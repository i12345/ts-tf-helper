import * as tf from "@tensorflow/tfjs"
import { TensorLike } from "@tensorflow/tfjs-core/dist/types"
import { castAsTensor, TensorOrTensorLike } from "../core"
import { getShapeRankIndex } from "./shape"

export interface RankAtOrAboveMap {
    [tf.Rank.R0]: tf.Rank.R0
    [tf.Rank.R1]: tf.Rank.R1 | RankAtOrAboveMap[tf.Rank.R0]
    [tf.Rank.R2]: tf.Rank.R2 | RankAtOrAboveMap[tf.Rank.R1]
    [tf.Rank.R3]: tf.Rank.R3 | RankAtOrAboveMap[tf.Rank.R2]
    [tf.Rank.R4]: tf.Rank.R4 | RankAtOrAboveMap[tf.Rank.R3]
    [tf.Rank.R5]: tf.Rank.R5 | RankAtOrAboveMap[tf.Rank.R4]
    [tf.Rank.R6]: tf.Rank.R6 | RankAtOrAboveMap[tf.Rank.R5]
}

export type RankAtOrAbove<Rank extends tf.Rank> = RankAtOrAboveMap[Rank]

export interface RankOneSmallerMap {
    [tf.Rank.R0]: never
    [tf.Rank.R1]: tf.Rank.R0
    [tf.Rank.R2]: tf.Rank.R1
    [tf.Rank.R3]: tf.Rank.R2
    [tf.Rank.R4]: tf.Rank.R3
    [tf.Rank.R5]: tf.Rank.R4
    [tf.Rank.R6]: tf.Rank.R5
}

export type RankOneSmaller<Rank extends tf.Rank> = RankOneSmallerMap[Rank]

export type RankIndex = 0 | 1 | 2 | 3 | 4 | 5 | 6

export function getRankIndex<Rank extends tf.Rank>(rank: Rank): RankIndex {
    switch(rank) {
        case tf.Rank.R0: return 0
        case tf.Rank.R1: return 1
        case tf.Rank.R2: return 2
        case tf.Rank.R3: return 3
        case tf.Rank.R4: return 4
        case tf.Rank.R5: return 5
        case tf.Rank.R6: return 6
        default: throw new Error("invalid rank")
    }
}

export function getRankByIndex<Rank extends tf.Rank>(rankIndex: RankIndex): Rank {
    switch(rankIndex) { 
        case 0: return <Rank>tf.Rank.R0
        case 1: return <Rank>tf.Rank.R1
        case 2: return <Rank>tf.Rank.R2
        case 3: return <Rank>tf.Rank.R3
        case 4: return <Rank>tf.Rank.R4
        case 5: return <Rank>tf.Rank.R5
        case 6: return <Rank>tf.Rank.R6
    }
}

export function getTensorOrTensorLikeRankIndex<Rank extends tf.Rank>(t: TensorOrTensorLike<Rank>): RankIndex {
    if(t instanceof tf.Tensor || castAsTensor(t).shape) {
        return getShapeRankIndex((<tf.Tensor>t).shape)
    }
    else {
        let tensorLike: TensorLike = t

        if(typeof tensorLike == 'number' ||
            typeof tensorLike == 'string' ||
            typeof tensorLike == 'boolean') {
            return 0
        }
        else {
            let rank = 0
            let tensorLikeArray: ArrayLike<TensorLike>
            while(
                    (tensorLike !== undefined) &&
                    (typeof tensorLike != 'string') &&
                    (typeof (tensorLikeArray = <ArrayLike<number>>tensorLike).length == 'number')
                ) {
                rank++
                tensorLike = tensorLikeArray[0]
            }

            return <RankIndex>rank
        }
    }
}