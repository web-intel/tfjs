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

import {backend_util} from '@tensorflow/tfjs-core';

import {conv2dCommonSnippet} from './conv2d_mm_webgpu';
import {makeMatMulLWSSource} from './matmul_lws_webgpu';
import {WebGPUProgram} from './webgpu_program';

const getMaxComponents = (size: number) => {
  if (size % 4 === 0) {
    return 4;
  } else if (size % 3 === 0) {
    return 3;
  } else if (size % 2 === 0) {
    return 2;
  }

  return 1;
};

export class Conv2DMMLWSProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  dispatchLayout: {x: number[], y: number[], z: number[]};
  dispatch: [number, number, number];
  variableNames = ['x', 'W'];
  variableComponents: number[];
  outputComponent: number;
  uniforms =
      `filterDims : vec2<i32>, pads : vec2<i32>, strides : vec2<i32>, dilations : vec2<i32>, dimAOuter : i32, dimBOuter : i32, dimInner : i32,`;
  workgroupSize: [number, number, number] = [64, 1, 1];
  addBias: boolean;
  activation: backend_util.Activation;
  hasPreluActivationWeights: boolean;
  isChannelsLast: boolean;
  fitAOuter: boolean;
  fitBOuter: boolean;
  fitInner: boolean;
  components: number;
  aComponents: number;
  outputNumber: number;
  tileN: number;
  tileM: number;
  tileK: number;

  constructor(
      convInfo: backend_util.Conv2DInfo, M: number, N: number, K: number,
      addBias = false, activation: backend_util.Activation = null,
      hasPreluActivationWeights = false) {
    this.outputShape = convInfo.outShape;
    this.isChannelsLast = convInfo.dataFormat === 'channelsLast';
    const batch = this.outputShape[0];
    this.dispatchLayout = {x: [3], y: [1, 2], z: [0]};
    // Support getMaxComponents when the bias and activation support it.
    this.components = N % 4 === 0 ? 4 : 1;
    this.aComponents = getMaxComponents(K);
    this.tileN = N < 32 ? N : 32;
    this.tileM = M < 32 ? M : 32;
    this.tileK = K < 32 ?
        K :
        (K > 1000 ? Math.ceil(32 / this.aComponents) * this.aComponents :
                    Math.ceil(16 / this.aComponents) * this.aComponents);
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
    this.dispatch = [numWorkgroups, 1, 1];

    this.outputComponent = this.components;
    this.variableComponents =
        [this.aComponents === 3 ? 1 : this.aComponents, this.components];
    if (addBias) {
      this.variableNames.push('bias');
      this.variableComponents.push(this.components);
    }

    if (hasPreluActivationWeights) {
      this.variableNames.push('preluActivationWeights');
      this.variableComponents.push(this.components);
    }

    this.addBias = addBias;
    this.activation = activation;
    this.hasPreluActivationWeights = hasPreluActivationWeights;
    this.fitAOuter = M % this.tileM === 0;
    this.fitBOuter = N % this.tileN === 0;
    this.fitInner = K % this.tileK === 0;

    this.shaderKey =
        `conv2DMMLWS_${this.activation}}_${this.fitAOuter}_${this.fitBOuter}_${
            this.fitInner}_${this.tileK}_${this.tileM}_${this.tileN}_${
            this.components}_${this.aComponents}_${this.outputNumber}`;
  }

  getUserCode(): string {
    const userCode = `
    ${
        conv2dCommonSnippet(
            this.isChannelsLast, this.fitAOuter, this.fitBOuter, this.fitInner,
            this.addBias, this.activation, this.hasPreluActivationWeights,
            this.aComponents, this.components, this.components)}
    ${
        makeMatMulLWSSource(
            this.workgroupSize[0], this.outputNumber, this.components,
            this.aComponents, this.tileM, this.tileN, this.tileK, true)}
  `;
    return userCode;
  }
}
