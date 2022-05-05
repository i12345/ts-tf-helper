import * as tf from '@tensorflow/tfjs'
import { TensorOrTensorLike } from '../core'
import { emptyShape, getShapeRank, getTensorOrTensorLikeShape } from '../helpers'
import { ensureIsTensor } from './ensure-tensor'

export function resize<Rank extends tf.Rank>(
        value: TensorOrTensorLike<Rank>,
        newShape: tf.ShapeMap[Rank]
    ): tf.Tensor<Rank> {
    if(newShape.length == 0) {
        return ensureIsTensor(value)
    }

    let paddings =
        <tf.ShapeMap[Rank]>
        getTensorOrTensorLikeShape(value)
            .map(
                    (dimenstionality, dimension) =>
                        newShape[dimension] - dimenstionality
                )
    
    return <tf.Tensor<Rank>>
        tf.pad(
                value,
                paddings.map(
                        padding =>
                            (padding > 0) ?
                                [0, padding] :
                                [0, 0]
                    )
            ).slice(
                    emptyShape(getShapeRank(newShape)),
                    newShape
                )
}