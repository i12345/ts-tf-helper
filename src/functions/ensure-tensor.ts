import * as tf from '@tensorflow/tfjs'
import { TensorLike1D, TensorLike2D, TensorLike3D, TensorLike4D, TensorLike5D } from '@tensorflow/tfjs-core/dist/types'
import { castAsTensor, TensorOrTensorLike } from "../core"
import { getRankByIndex, getTensorOrTensorLikeRankIndex } from "../helpers"

export function ensureIsTensor<Rank extends tf.Rank>(
        t: TensorOrTensorLike<Rank>,
        rank: Rank = getRankByIndex(getTensorOrTensorLikeRankIndex(t))
    ): tf.Tensor<Rank> {
    if(t instanceof tf.Tensor || castAsTensor(t).rank) {
        return castAsTensor(t)
    }

    switch(rank) {
        case tf.Rank.R0: return <tf.Tensor<Rank>><unknown>tf.scalar(<tf.ScalarLike>t)
        case tf.Rank.R1: return <tf.Tensor<Rank>><unknown>tf.tensor1d(<TensorLike1D>t)
        case tf.Rank.R2: return <tf.Tensor<Rank>><unknown>tf.tensor2d(<TensorLike2D>t)
        case tf.Rank.R3: return <tf.Tensor<Rank>><unknown>tf.tensor3d(<TensorLike3D>t)
        case tf.Rank.R4: return <tf.Tensor<Rank>><unknown>tf.tensor4d(<TensorLike4D>t)
        case tf.Rank.R5: return <tf.Tensor<Rank>><unknown>tf.tensor5d(<TensorLike5D>t)
        case tf.Rank.R6: throw new Error("Cannot make 6D tensors")
        default: throw new Error("invalid rank")
    }
}