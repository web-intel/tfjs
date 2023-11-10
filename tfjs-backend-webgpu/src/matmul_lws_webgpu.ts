/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {backend_util, TensorInfo} from '@tensorflow/tfjs-core';

import {activationFnSnippet} from './activation_util';
import {matMulReadWriteFnSource} from './matmul_packed_webgpu';
import {getMainHeaderString as main, typeSnippet, WebGPUProgram} from './webgpu_program';

export function makeMatMulLWSSource(
    workgroupSize: number, outputNumber: number, components: number,
    aComponents: number, tileM: number, tileN: number, tileK: number,
    isConv = false): string {
  const aValueType = typeSnippet(aComponents);
  const bValueType = typeSnippet(components);
  const outValueType = typeSnippet(components);
  // const paddedVirtualM = Math.ceil(M / this.tileM);
  // const paddedvirtualN = Math.ceil(N / this.tileN);
  const calcResult = (): string => {
    let calcStr = `var aData: ${aValueType};`;
    for (let i = 0; i < aComponents; i++) {
      calcStr += `
            let bData${i} = mm_Bsub[k + ${i}][localCol];`;
    }
    for (let i = 0; i < outputNumber; i++) {
      calcStr += `aData = mm_Asub[localRow + ${i}][k / ${aComponents}];`;

      for (let j = 0; j < aComponents; j++) {
        calcStr += `
          values[${i}] = fma(${bValueType}(aData${
            aComponents === 1 ? '' : `[${j}]`}), bData${j}, values[${i}]);\n`;
      }
    }
    return calcStr;
  };

  return `
  var<workgroup> mm_Asub: array<array<${aValueType}, ${tileK / aComponents}>, ${
      tileM}>;
  var<workgroup> mm_Bsub: array<array<${bValueType}, ${tileN / components}>, ${
      tileK}>;
  ${main()} {
    let virtualGlobalId = workgroupId.z * numWorkgroups.x * numWorkgroups.y +
        workgroupId.y * numWorkgroups.x + workgroupId.x;
    let stride0 = (u32(uniforms.dimBOuter) - 1u) / ${tileN}u + 1u;
    let tileColStart = (virtualGlobalId % stride0) * ${tileN}u;
    var index1 = virtualGlobalId / stride0;
    let stride1 = (u32(uniforms.dimAOuter) - 1u) / ${tileM}u + 1u;
    let tileRowStart = (index1 % stride1) * ${tileM}u;
    let batch = index1 / stride1;
    let batchA = ${isConv ? 'i32(batch)' : 'i32(batch) % uniforms.aShape[0]'};
    let batchB = ${isConv ? '0i' : 'i32(batch) % uniforms.bShape[0]'};

    var values: array<${outValueType}, ${outputNumber}>;
    let numTiles = (u32(uniforms.dimInner) - 1u) / ${tileK} + 1u;
    var kStart = 0u;
    // Loop over shared dimension.
    for (var t = 0u; t < numTiles; t = t + 1u) {
      // Load one tile of A into local memory.
      for (var tIndex = localIndex; tIndex < ${
      tileM * tileK / aComponents}; tIndex += ${workgroupSize}) {
          let inputRow = tIndex / ${tileK / aComponents};
          let inputCol = tIndex % ${tileK / aComponents};

          mm_Asub[inputRow][inputCol] = mm_readA(batchA, i32(tileRowStart + inputRow), i32(kStart + inputCol * ${
      aComponents}));
      }

      // Load one tile of B into local memory.
      for (var tIndex = localIndex; tIndex < ${
      tileK * tileN / components}; tIndex += ${workgroupSize}) {
          let inputRow = tIndex / ${tileN / components};
          let inputCol = tIndex % ${tileN / components};
          mm_Bsub[inputRow][inputCol] = mm_readB(batchB, i32(kStart + inputRow), i32(tileColStart + inputCol * ${
      components}));
      }
      kStart = kStart + ${tileK};
      workgroupBarrier();

      // Compute values for a single thread.
      for (var k = 0; k < ${tileK}; k = k + ${aComponents}) {
        for (var tIndex = localIndex; tIndex < ${
      (tileM / outputNumber) *
      (tileN / components)}; tIndex += ${workgroupSize}) {
          let localRow = tIndex / ${tileN / components} * ${outputNumber};
          let localCol = tIndex % ${tileN / components};
        ${calcResult()}
      }
      }
      workgroupBarrier();
    }

    for (var tIndex = localIndex; tIndex < ${
      (tileM / outputNumber) *
      (tileN / components)}; tIndex += ${workgroupSize}) {
      let localRow = tIndex / ${tileN / components} * ${outputNumber};
      let localCol = tIndex % ${tileN / components};
      let globalCol = tileColStart + localCol * ${components};
      let globalRow = tileRowStart + localRow;
      for (var i = 0u; i < ${outputNumber}u; i++) {
        mm_write(i32(batch), i32(globalRow + i), i32(globalCol), values[i]);
      }
    }
  }
  `;
}

const getMaxComponents = (size: number) => {
  if (size % 4 === 0) {
    return 4;
  } else if (size % 2 === 0) {
    return 2;
  }

  return 1;
};

export class MatMulLWSProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  variableNames = ['A', 'B'];
  variableComponents: number[];
  outputComponent: number;
  uniforms = `dimAOuter : i32, dimBOuter : i32, dimInner : i32,`;
  workgroupSize: [number, number, number] = [64, 1, 1];
  transposeA: boolean;
  transposeB: boolean;
  addBias: boolean;
  activation: backend_util.Activation;
  hasPreluActivationWeights: boolean;
  components: number;
  aComponents: number;
  outputNumber: number;
  fitAOuter: boolean;
  fitBOuter: boolean;
  fitInner: boolean;
  tileN: number;
  tileM: number;
  tileK: number;

  constructor(
      outputShape: [number, number, number], sharedDim: number,
      transposeA = false, transposeB = false, bias: TensorInfo = null,
      activation: backend_util.Activation = null,
      preluActivationWeights: TensorInfo = null) {
    this.outputShape = outputShape;
    const batch = outputShape[0];
    const M = outputShape[1];
    const N = outputShape[2];
    const K = sharedDim;
    // Support getMaxComponents when the bias and activation support it.
    this.components = N % 4 === 0 ? 4 : 1;
    this.aComponents = getMaxComponents(K);
    this.tileN = N < 32 ? N : 32;
    this.tileM = M < 32 ? M : 32;
    this.tileK = K < 32 ? K : (K > 1000 ? 32 : 16);
    // The output number of each thread.
    this.outputNumber =
        this.tileM < 4 ? this.tileM : getMaxComponents(this.tileM);
    if (this.tileN % this.components !== 0 ||
        this.tileM % this.outputNumber !== 0 ||
        this.tileK % this.aComponents !== 0) {
      throw new Error(`tileN(${this.tileN}) must be divisible by components(${
          this.components}), tileM(${
          this.tileM}) must be divisible by outputNumber(${
          this.outputNumber}), tileK(${
          this.tileK}) must be divisible by aComponents(${this.aComponents})`);
    }

    const numWorkgroups =
        batch * Math.ceil(M / this.tileM) * Math.ceil(N / this.tileN);
    this.dispatchLayout = {x: [], y: [1, 2], z: [0]};
    this.dispatch = [numWorkgroups, 1, 1];

    this.outputComponent = this.components;
    this.variableComponents = [this.aComponents, this.components];
    const addBias = bias != null;
    const hasPreluActivationWeights = preluActivationWeights != null;
    if (addBias) {
      this.variableNames.push('bias');
      this.variableComponents.push(this.components);
    }

    if (hasPreluActivationWeights) {
      this.variableNames.push('preluActivationWeights');
      this.variableComponents.push(this.components);
    }

    this.transposeA = transposeA;
    this.transposeB = transposeB;
    this.addBias = addBias;
    this.activation = activation;
    this.hasPreluActivationWeights = hasPreluActivationWeights;
    this.fitAOuter = M % this.tileM === 0;
    this.fitBOuter = N % this.tileN === 0;
    this.fitInner = K % this.tileK === 0;
    this.shaderKey = `matMulLWS_${this.activation}_${transposeA}_${
        transposeB}_${this.fitAOuter}_${this.fitBOuter}_${this.fitInner}_${
        this.tileK}_${this.tileM}_${this.tileN}_${this.components}_${
        this.aComponents}_${this.outputNumber}`;
  }

  getUserCode(): string {
    const userCode = `
      ${
        activationFnSnippet(
            this.activation, this.hasPreluActivationWeights,
            this.components === 4)}
      ${
        matMulReadWriteFnSource(
            this.addBias, this.activation, this.transposeA, this.transposeB,
            this.fitAOuter, this.fitBOuter, this.fitInner, this.components,
            this.aComponents)}
      ${
        makeMatMulLWSSource(
            this.workgroupSize[0], this.outputNumber, this.components,
            this.aComponents, this.tileM, this.tileN, this.tileK)}
    `;
    return userCode;
  }
}
