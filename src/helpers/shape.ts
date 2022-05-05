import * as tf from '@tensorflow/tfjs'
import { castAsTensor, TensorOrTensorLike } from '../core'
import { getRankByIndex, RankIndex } from './rank'

export function getShapeRankIndex<Rank extends tf.Rank>(shape: tf.ShapeMap[Rank]): RankIndex {
    return <RankIndex>shape.length
}

export function getShapeRank<Rank extends tf.Rank>(shape: tf.ShapeMap[Rank]): Rank {
    return getRankByIndex(<RankIndex>shape.length)
}

export function emptyShape<Rank extends tf.Rank>(rank: Rank): tf.ShapeMap[Rank] {
    switch(rank) {
        case tf.Rank.R0: return <tf.ShapeMap[Rank]><number[]>[]
        case tf.Rank.R1: return <tf.ShapeMap[Rank]>[0]
        case tf.Rank.R2: return <tf.ShapeMap[Rank]>[0, 0]
        case tf.Rank.R3: return <tf.ShapeMap[Rank]>[0, 0, 0]
        case tf.Rank.R4: return <tf.ShapeMap[Rank]>[0, 0, 0, 0]
        case tf.Rank.R5: return <tf.ShapeMap[Rank]>[0, 0, 0, 0, 0]
        case tf.Rank.R6: return <tf.ShapeMap[Rank]>[0, 0, 0, 0, 0, 0]
        default: throw new Error("invalid rank")
    }
}

export function getTensorOrTensorLikeShape<Rank extends tf.Rank>(t: TensorOrTensorLike<Rank>): tf.ShapeMap[Rank] {
    if(t instanceof tf.Tensor || castAsTensor(t).shape) {
        return castAsTensor(t).shape
    }
    else {
        let tensorLike: tf.TensorLike = t

        if(typeof tensorLike == 'number' ||
            typeof tensorLike == 'string' ||
            typeof tensorLike == 'boolean') {
            return <tf.ShapeMap[Rank]><number[]>[]
        }
        else {
            let shape: number[] = []

            let tensorLikeArray: ArrayLike<tf.TensorLike>
            while(
                    (tensorLike !== undefined) &&
                    (typeof tensorLike != 'string') &&
                    (typeof (tensorLikeArray = <ArrayLike<number>>tensorLike).length == 'number')
                ) {
                shape.push(tensorLikeArray.length)
                tensorLike = tensorLikeArray[0]
            }

            return <tf.ShapeMap[Rank]>shape
        }
    }
}