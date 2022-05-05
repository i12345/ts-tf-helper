import * as tf from '@tensorflow/tfjs'

export function project(
        vector: tf.Tensor1D,
        projectionMatrix: tf.Tensor2D
    ): tf.Tensor1D {
    return vector.expandDims(1).mul(projectionMatrix).sum(0)
}