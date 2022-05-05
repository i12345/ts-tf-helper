import * as tf from '@tensorflow/tfjs'
import { TensorOrTensorLike } from '../core'
import { resize, ensureIsTensor } from '../functions'

export class ResizeableTensorVariable<
        Rank extends tf.Rank,
        DataType extends tf.DataType = 'float32'
    > {
    private _variable: tf.Variable<Rank>
    private _shape: tf.ShapeMap[Rank]

    get capacity(): tf.ShapeMap[Rank] {
        return this._variable.shape
    }

    set capacity(value: tf.ShapeMap[Rank]) {
        this._variable = tf.variable(resize(this._variable, value))
    }

    get shape(): tf.ShapeMap[Rank] {
        return this.shape
    }

    set shape(value: tf.ShapeMap[Rank]) {
        let size_difference = value.map((size_i, i) => this.capacity[i] - size_i)
        
        if(!size_difference.every(size => size >= 0)) {
            this.capacity =
                <tf.ShapeMap[Rank]>
                this.capacity.map(
                        (capacity, i) =>
                            (size_difference[i] < 0) ?
                                capacity + (-size_difference) :
                                capacity
                    )
        }

        this.shape = value
    }
    
    get value(): tf.Tensor<Rank> {
        return resize(this._variable, this.shape)
    }

    set value(value: TensorOrTensorLike<Rank>) {
        this._variable.assign(resize(value, this.capacity))
    }

    constructor(
            initialValue: TensorOrTensorLike<Rank>,
            dtype?: DataType
        ) {
        this._variable = tf.variable(ensureIsTensor(initialValue), false, undefined, dtype)
        this._shape = this._variable.shape
    }
}