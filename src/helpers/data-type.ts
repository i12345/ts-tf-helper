import * as tf from '@tensorflow/tfjs'
import { castAsTensor, TensorOrTensorLike } from '../core'

export function getTensorOrTensorLikeDataType<Rank extends tf.Rank>(t: TensorOrTensorLike<Rank>): tf.DataType {
    if(t instanceof tf.Tensor || castAsTensor(t).dtype) {
        return castAsTensor(t).dtype
    }
    else {
        let tensorLike: tf.TensorLike = t
        let tensorLikeArray: ArrayLike<tf.TensorLike>
        while(
                (tensorLike !== undefined) &&
                (typeof tensorLike != 'string') &&
                (typeof (tensorLikeArray = <ArrayLike<number>>tensorLike).length == 'number')
            ) {
            if(tensorLikeArray instanceof Int8Array ||
                tensorLikeArray instanceof Int16Array ||
                tensorLikeArray instanceof Int32Array ||
                tensorLikeArray instanceof BigInt64Array ||
                tensorLikeArray instanceof Uint8ClampedArray ||
                tensorLikeArray instanceof Uint8Array ||
                tensorLikeArray instanceof Uint16Array ||
                tensorLikeArray instanceof Uint32Array ||
                tensorLikeArray instanceof BigUint64Array) {
                return 'int32'
            }

            tensorLike = tensorLikeArray[0]
        }

        switch(typeof tensorLike) {
            case 'string': return 'string'
            case 'number': return 'float32'
            case 'boolean': return 'bool'
            default: throw new Error("invalid dtype")
        }
    }
}