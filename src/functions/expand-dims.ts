import * as tf from '@tensorflow/tfjs'
import { TensorOrTensorLike } from '../core';
import { getTensorOrTensorLikeRankIndex, getRankIndex, getShapeRankIndex, RankAtOrAbove } from '../helpers';

export function expandDimsTo<SrcRank extends tf.Rank, DstRank extends RankAtOrAbove<SrcRank>>(
    src: TensorOrTensorLike<SrcRank>,
    dstRankOrShapeExample: DstRank | tf.ShapeMap[DstRank]
) {
    let srcRankIndex = getTensorOrTensorLikeRankIndex(src)
    let dstRankIndex =
        (typeof dstRankOrShapeExample == typeof tf.Rank) ?
            getRankIndex(<DstRank>dstRankOrShapeExample) :
            getShapeRankIndex(<tf.ShapeMap[DstRank]>dstRankOrShapeExample)

    let t: tf.Tensor | tf.TensorLike = src;
    for(let i = dstRankIndex - srcRankIndex; i > 0; i--) {
        t = tf.expandDims(t)
    }

    return <TensorOrTensorLike<DstRank>>t
}