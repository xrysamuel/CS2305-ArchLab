package GEMM

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import scala.util.Random

class GEMMTest extends AnyFlatSpec with ChiselScalatestTester {

  "GeneralMatrixMultiplicationModule test" should "pass" in {
    val test_cfg = MatMulConfig()
    val nmk_list = Array((6, 7, 3), (16, 1, 11), (2, 7, 16))
    test(new GEMM_TOP(cfg = test_cfg)) { dut =>
      dut.clock.setTimeout(1000000)
      for ((n, m, k) <- nmk_list) {
        dut.clock.step(1)
        // data preparision & poke
        println(s"Begin test. Matrix size: ${n} ${m} ${k}")
        val MatA = generateRandomMatrixSigned(
          rows = n,
          cols = m,
          bits = test_cfg.vecDWidth
        )
        val MatB = generateRandomMatrixSigned(
          rows = m,
          cols = k,
          bits = test_cfg.vecDWidth
        )
        println("Calculate MatC.")
        val MatC = matrixMultiplySigned(A = MatA, B = MatB)

        // input A to mem
        println("Loading MatA..")
        dut.io.op.poke(test_cfg.OP_ACCESS_MEM)
        for (row <- 0 until n) {
          println("")
          for (col <- 0 until m) {
            val addr_in_mem =
              test_cfg.matA_baseAddr + row * test_cfg.gemm_matsize + col
            dut.io.dataIn.poke(MatA(row)(col))
            print(s"${MatA(row)(col)}\t")
            dut.io.addr.poke(addr_in_mem.U)
            dut.io.writeEn.poke(true.B)
            dut.io.enable.poke(true.B)
            dut.clock.step(1)
          }
        }

        // input B to mem
        println("\nLoading MatB..")
        for (row <- 0 until m) {
          println("")
          for (col <- 0 until k) {
            val addr_in_mem =
              test_cfg.matB_baseAddr + row * test_cfg.gemm_matsize + col
            print(s"${MatB(row)(col)}\t")
            dut.io.dataIn.poke(MatB(row)(col))
            dut.io.addr.poke(addr_in_mem.U)
            dut.io.writeEn.poke(true.B)
            dut.io.enable.poke(true.B)
            dut.clock.step(1)
          }
        }

        dut.io.writeEn.poke(false.B)
        dut.io.enable.poke(false.B)

        // compute
        println("Begin Compute..")
        dut.io.op.poke(test_cfg.OP_COMPUTE)
        dut.io.start.poke(true)
        dut.io.N.poke(n)
        dut.io.M.poke(m)
        dut.io.K.poke(k)
        dut.clock.step(1)
        dut.io.start.poke(false)
        while (dut.io.busy.peekBoolean()) {
          dut.clock.step(1)
        }
        println("Complete.")

        // store and check C'res
        dut.io.op.poke(test_cfg.OP_ACCESS_MEM)
        for (row <- 0 until n) {
          println("")
          for (col <- 0 until k) {
            val addr_in_mem =
              test_cfg.matC_baseAddr + row * test_cfg.gemm_matsize + col
            dut.io.addr.poke(addr_in_mem.U)
            dut.io.writeEn.poke(false.B)
            dut.io.enable.poke(true.B)
            dut.clock.step(1)
            dut.io.dataOut.expect(MatC(row)(col))
            print(s"(${MatC(row)(col)})${dut.io.dataOut.peekInt()}\t")
          }
        }
      }

    }
  }

  def generateRandomMatrixSigned(
      rows: Int,
      cols: Int,
      bits: Int
  ): Array[Array[Int]] = {
    val maxValue = math.pow(2, bits - 1).toInt
    Array.fill(rows, cols)((Random.nextInt(maxValue) - maxValue))
  }

  def matrixMultiplySigned(
      A: Array[Array[Int]],
      B: Array[Array[Int]]
  ): Array[Array[Int]] = {
    require(
      A(0).length == B.length,
      "The number of columns in A is not equal to the number of rows in B."
    )

    val result = Array.ofDim[Int](A.length, B(0).length)
    for (i <- A.indices) {
      for (j <- B(0).indices) {
        for (k <- A(0).indices) {
          result(i)(j) += A(i)(k) * B(k)(j)
        }
      }
    }
    result
  }

}
