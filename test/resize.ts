import { suite, test } from '@testdeck/mocha'
import { assert } from 'chai'
import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-node'
import { ensureIsTensor, getTensorOrTensorLikeShape, resize, TensorLikeMap, TensorOrTensorLike } from '../src'

@suite
class ResizeTests {
    @test
    resize_0_0() {
        this.run4(
                123,
                123
            )
    }

    @test
    resize_1_0() {
        this.run4(
                [100, 200, 300, 400],
                [100, 200]
            )
    }

    @test
    resize_1_1() {
        this.run4<tf.Rank.R1>(
                [100, 200, 300, 400],
                []
            )
    }

    @test
    resize_1_2() {
        this.run4(
                [100, 200, 300, 400],
                [100, 200, 300, 400]
            )
    }

    @test
    resize_1_3() {
        this.run4(
                [100, 200, 300, 400],
                [100, 200, 300, 400, 0, 0, 0, 0, 0, 0]
            )
    }

    @test
    resize_2_0() {
        this.run4(
                [
                    [10, 20, 30, 40, 50],
                    [60, 70, 80, 90, 100],
                    [-10, -20, -30, -40, -50],
                    [-60, -70, -80, -90, -100]
                ],
                [
                    [10, 20, 30],
                    [60, 70, 80],
                    [-10, -20, -30],
                    [-60, -70, -80]
                ]
            )
    }

    @test
    resize_2_1() {
        this.run4(
                [
                    [10, 20, 30, 40, 50],
                    [60, 70, 80, 90, 100],
                    [-10, -20, -30, -40, -50],
                    [-60, -70, -80, -90, -100]
                ],
                [
                    [10, 20, 30]
                ]
            )
    }

    @test
    resize_2_2() {
        this.run4(
                [
                    [10, 20, 30, 40, 50],
                    [60, 70, 80, 90, 100],
                    [-10, -20, -30, -40, -50],
                    [-60, -70, -80, -90, -100]
                ],
                [
                    []
                ]
            )
    }

    @test
    resize_2_3() {
        this.run4(
                [
                    [10, 20, 30, 40, 50],
                    [60, 70, 80, 90, 100],
                    [-10, -20, -30, -40, -50],
                    [-60, -70, -80, -90, -100]
                ],
                [
                    [10, 20, 30, 40, 50],
                    [60, 70, 80, 90, 100],
                    [-10, -20, -30, -40, -50],
                    [-60, -70, -80, -90, -100]
                ]
            )
    }

    @test
    resize_2_4() {
        this.run4(
                [
                    [10, 20, 30, 40, 50],
                    [60, 70, 80, 90, 100],
                    [-10, -20, -30, -40, -50],
                    [-60, -70, -80, -90, -100]
                ],
                [
                    [10, 20, 30, 40, 50]
                ]
            )
    }

    @test
    resize_3_0() {
        this.run4(
                [
                    [
                        [1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]
                    ],
                    [
                        [10, 20, 30],
                        [40, 50, 60],
                        [70, 80, 90]
                    ]
                ],
                [
                    [
                        [1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]
                    ],
                    [
                        [10, 20, 30],
                        [40, 50, 60],
                        [70, 80, 90]
                    ]
                ]
            )
    }

    @test
    resize_3_1() {
        this.run4(
                [
                    [
                        [1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]
                    ],
                    [
                        [10, 20, 30],
                        [40, 50, 60],
                        [70, 80, 90]
                    ]
                ],
                [
                    [
                        [1, 2, 3],
                        [4, 5, 6]
                    ],
                    [
                        [10, 20, 30],
                        [40, 50, 60]
                    ]
                ]
            )
    }

    @test
    resize_3_2() {
        this.run4(
                [
                    [
                        [1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]
                    ],
                    [
                        [10, 20, 30],
                        [40, 50, 60],
                        [70, 80, 90]
                    ]
                ],
                [
                    [
                        [1, 2],
                        [4, 5],
                        [7, 8]
                    ],
                    [
                        [10, 20],
                        [40, 50],
                        [70, 80]
                    ]
                ]
            )
    }

    @test
    resize_3_3() {
        this.run4(
                [
                    [
                        [1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]
                    ],
                    [
                        [10, 20, 30],
                        [40, 50, 60],
                        [70, 80, 90]
                    ]
                ],
                [
                    [
                        [1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]
                    ]
                ]
            )
    }

    @test
    resize_3_4() {
        this.run4(
                [
                    [
                        [1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]
                    ],
                    [
                        [10, 20, 30],
                        [40, 50, 60],
                        [70, 80, 90]
                    ]
                ],
                [
                    [
                        [1, 2],
                        [4, 5]
                    ],
                    [
                        [10, 20],
                        [40, 50]
                    ]
                ]
            )
    }

    @test
    resize_3_5() {
        this.run4(
                [
                    [
                        [1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]
                    ],
                    [
                        [10, 20, 30],
                        [40, 50, 60],
                        [70, 80, 90]
                    ]
                ],
                [
                    [
                        [1, 2],
                        [4, 5]
                    ]
                ]
            )
    }

    @test
    resize_3_6() {
        this.run4(
                [
                    [
                        [1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]
                    ],
                    [
                        [10, 20, 30],
                        [40, 50, 60],
                        [70, 80, 90]
                    ]
                ],
                [
                    [
                        [1]
                    ]
                ]
            )
    }

    @test
    resize_3_7() {
        this.run4(
                [
                    [
                        [1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]
                    ],
                    [
                        [10, 20, 30],
                        [40, 50, 60],
                        [70, 80, 90]
                    ]
                ],
                [
                    [
                        []
                    ]
                ]
            )
    }

    run4<Rank extends tf.Rank>(
            original: TensorLikeMap[Rank],
            final: TensorLikeMap[Rank]
        ) {
        const original_tensor = ensureIsTensor(original)
        const final_tensor = ensureIsTensor(final)

        this.run(original, final)
        this.run(original_tensor, final)
        this.run(original, final_tensor)
        this.run(original_tensor, final_tensor)
    }

    run<Rank extends tf.Rank>(
            original: TensorOrTensorLike<Rank>,
            final: TensorOrTensorLike<Rank>
        ) {
        const resized = resize(original, getTensorOrTensorLikeShape(final))
        const equalsData = resized.equal(final).dataSync()
        assert.isTrue(equalsData.length == 0 || equalsData[0] == tf.scalar(true).dataSync()[0])
    }
}