import * as tf from "@tensorflow/tfjs"
import { ScalarLike, TensorLike1D, TensorLike2D, TensorLike3D, TensorLike4D, TensorLike5D, TensorLike6D } from "@tensorflow/tfjs-core/dist/types"

export interface TensorMap {
    [tf.Rank.R0]: tf.Scalar
    [tf.Rank.R1]: tf.Tensor1D
    [tf.Rank.R2]: tf.Tensor2D
    [tf.Rank.R3]: tf.Tensor3D
    [tf.Rank.R4]: tf.Tensor4D
    [tf.Rank.R5]: tf.Tensor5D
    [tf.Rank.R6]: never
}

export interface TensorLikeMap {
    [tf.Rank.R0]: ScalarLike
    [tf.Rank.R1]: TensorLike1D
    [tf.Rank.R2]: TensorLike2D
    [tf.Rank.R3]: TensorLike3D
    [tf.Rank.R4]: TensorLike4D
    [tf.Rank.R5]: TensorLike5D
    [tf.Rank.R6]: TensorLike6D
}

export type TensorOrTensorLike<Rank extends tf.Rank = tf.Rank.R1> = tf.Tensor<Rank> | TensorLikeMap[Rank]

export type ScalarOrScalarLike = TensorOrTensorLike<tf.Rank.R0>

export function castAsTensor<Rank extends tf.Rank>(t: TensorOrTensorLike<Rank>): tf.Tensor<Rank> {
    return <tf.Tensor<Rank>><unknown>t
}