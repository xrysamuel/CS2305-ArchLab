package GEMM

import chisel3._
import chisel3.util._

case class MatMulConfig(
    vecDWidth: Int = 8, // Precision of matrices A and B (8 bits)
    resDWidth: Int = 32, // Precision of result matrix C (32 bits)
    matSize: Int = 4, // Size of the matrices (16x16)
    gemm_matsize: Int = 8, // Size of the matrices for GEMM (32x32)
    SEL_A: UInt = 0.U, // Operation selection for matrix A
    SEL_B: UInt = 1.U, // Operation selection for matrix B
    SEL_C: UInt = 2.U, // Operation selection for matrix C
    OP_ACCESS_MEM: UInt = 1.U, // Operation selection for the top module - accessing memory
    OP_COMPUTE: UInt = 0.U // Operation selection for the top module - performing computation
) {

  val gemm_elem_num = gemm_matsize * gemm_matsize // Size of matrices A, B, and C for computation
  val memory_size = gemm_elem_num * 3 // The memory module needs to be large enough to accommodate matrices A, B, and C. Here, we assume each matrix element occupies 32 bits in memory.
  val matA_baseAddr = 0 // Base address for storing matrix A in memory
  val matB_baseAddr = gemm_elem_num // Base address for storing matrix B in memory
  val matC_baseAddr = 2 * gemm_elem_num // Base address for storing matrix C in memory
  val memory_width = resDWidth // The number of bits corresponding to each address in memory

  val elem_num = matSize * matSize
}

class GEMM_TOP(cfg: MatMulConfig) extends Module {
  val io = IO(new Bundle {
    // ******** Load Data to Memory **********
    val dataIn = Input(SInt(cfg.memory_width.W))
    val dataOut = Output(SInt(cfg.memory_width.W))
    val addr = Input(UInt(log2Ceil(cfg.memory_size).W))
    val writeEn = Input(Bool())
    val enable = Input(Bool())

    // operation: access memory or compute
    val op = Input(UInt(2.W))

    // start mm module
    val start = Input(Bool())

    // compute finish
    val busy = Output(Bool())
  })

  val memory_module = Module(new MatMem(cfg.memory_size, cfg.memory_width))

  val mat_module = Module(new MatMulModule(cfg))

  // ******** Connection **********
  val mat_mul_access_mem = (io.op =/= cfg.OP_COMPUTE)
  memory_module.io.addr := Mux(mat_mul_access_mem, io.addr, mat_module.io.addr)
  memory_module.io.writeEn := Mux(mat_mul_access_mem, io.writeEn, mat_module.io.writeEn)
  memory_module.io.en := Mux(mat_mul_access_mem, io.enable, mat_module.io.enable)
  memory_module.io.dataIn := Mux(mat_mul_access_mem, io.dataIn, mat_module.io.dataOut)

  mat_module.io.dataIn := memory_module.io.dataOut
  mat_module.io.start := io.start
  io.dataOut := memory_module.io.dataOut.asSInt
  io.busy := mat_module.io.busy
}

class MatMem(MEM_ROW: Int, MEM_WIDTH: Int) extends Module {
  val io = IO(new Bundle {
    val addr = Input(UInt(log2Ceil(MEM_ROW).W))
    val writeEn = Input(Bool())
    val en = Input(Bool())
    val dataIn = Input(SInt(MEM_WIDTH.W))
    val dataOut = Output(SInt(MEM_WIDTH.W))
  })

  val mem = SyncReadMem(MEM_ROW, SInt(MEM_WIDTH.W))

  // single port
  when(io.writeEn & io.en) {
    mem.write(io.addr, io.dataIn)
  }
  io.dataOut := mem.read(io.addr, io.en && (!io.writeEn))
}

// **********************************************************************

class MatMulModule(cfg: MatMulConfig) extends Module {
  val io = IO(new Bundle {
    val addr = Output(UInt(log2Ceil(cfg.memory_size).W))
    val writeEn = Output(Bool())
    val enable = Output(Bool())
    val dataIn = Input(SInt(cfg.memory_width.W))
    val dataOut = Output(SInt(cfg.memory_width.W))

    val start = Input(Bool())
    val busy = Output(Bool())
  })


}

class MatMulScheduler(cfg: MatMulConfig) extends Module {
  val io = IO(new Bundle {
    val start = Input(Bool())
    val busy = Output(Bool())

  })

}

class TileMulModule(cfg: MatMulConfig) extends Module {
  val io = IO(new Bundle {
    val addr = Output(UInt(log2Ceil(cfg.memory_size).W))
    val writeEn = Output(Bool())
    val enable = Output(Bool())
    val dataIn = Input(SInt(cfg.memory_width.W))
    val dataOut = Output(SInt(cfg.memory_width.W))

    val iStart = Input(UInt(log2Ceil(cfg.gemm_matsize).W))
    val kStart = Input(UInt(log2Ceil(cfg.gemm_matsize).W))
    val iMask = Input(UInt(log2Ceil(cfg.matSize).W))
    val kMask = Input(UInt(log2Ceil(cfg.matSize).W))
    val jMax = Input(UInt(log2Ceil(cfg.gemm_matsize).W))

    val start = Input(Bool())
    val flush = Input(Bool())
    val busy = Output(Bool())
  })
  /*
   * TileMulModule: C[tile_i][tile_k] = The sum of A[tile_i][tile_j] * B[tile_j][tile_k] over tile_j
   * SA: C[tile_i][tile_k] += A[tile_i][tile_j] * B[tile_j][tile_k]
   *
   * 0: if start && !flush
   *        jump 1 ()
   *    else 
   *        jump 0 ()
   * 1: counter = 0 (stall, busy)
   * 2: aIn[:] = 0, bIn[:] = 0, index = max(counter - jMax, 0) (busy)
   * 3: dataOut = A[iStart+index][counter-index], jump 5 (stall, busy)
   * 4: dataOut = A[iStart+index][counter-index], bIn[index-1] = dataOut (stall, busy)
   * 5: dataOut = B[counter-index][kStart+index], aIn[index] = dataOut (stall, busy)
   *    if index = min(counter, matSize - 1)
   *        if counter == matSize - 1 + jMax
   *            jump 7 (stall, busy)
   *        else
   *            counter++, jump 6 (stall, busy)
   *    else
   *        index++, jump 4 (stall, busy)
   * 6: bIn[index] = dataOut, jump 2 (stall, busy)
   * 7: bIn[index] = dataOut, counter = 0, jump 8 (stall, busy)
   * 8: if counter == matSize
   *        jump 0 (busy)
   *    else
   *        aIn[:] = 0, bIn[:] = 0, counter++, jump 8 (busy)
   */

  val sa = Module(new SA(cfg.matSize, cfg.vecDWidth, cfg.resDWidth))
  val aIn = Seq.fill(cfg.matSize)(RegInit(0.S(cfg.vecDWidth.W)))
  val bIn = Seq.fill(cfg.matSize)(RegInit(0.S(cfg.vecDWidth.W)))
  sa.io.flush := false.B
  sa.io.stall := true.B
  for (i <- 0 until cfg.matSize) {
    sa.io.inHorizontal(i) := Mux(iMask < i.U, 0.U, aIn(i))
    sa.io.inVertical(i) := Mux(kMask < i.U, 0.U, bIn(i))
  }


}

class SA(val matSize: Int = 4, val width: Int = 8, val resDWidth: Int = 32) extends Module {
  val io = IO(new Bundle {
    val flush = Input(Bool())
    val stall = Input(Bool())
    val inHorizontal = Input(Vec(matSize, SInt(width.W)))
    val inVertical = Input(Vec(matSize, SInt(width.W)))
    val out = Output(Vec(matSize * matSize, SInt(resDWidth.W)))
  })

  val cells = Seq.fill(matSize, matSize)(Module(new SACell(width)))

  val wireHorizontal = Seq.fill(matSize, matSize - 1)(Wire(SInt(width.W)))
  val wireVertical = Seq.fill(matSize - 1, matSize)(Wire(SInt(width.W)))

  for (row <- 0 until matSize) {
    for (col <- 0 until matSize) {

      io.out(row * matSize + col) := cells(row)(col).io.out

      if (col < matSize - 1)
        wireHorizontal(row)(col) := cells(row)(col).io.outHorizontal
      if (col == 0)
        cells(row)(col).io.inHorizontal := io.inHorizontal(row)
      else
        cells(row)(col).io.inHorizontal := wireHorizontal(row)(col - 1)

      if (row < matSize - 1)
        wireVertical(row)(col) := cells(row)(col).io.outVertical
      if (row == 0)
        cells(row)(col).io.inVertical := io.inVertical(col)
      else
        cells(row)(col).io.inVertical := wireVertical(row - 1)(col)
    }
  }

  for (row <- 0 until matSize) {
    for (col <- 0 until matSize) {
      cells(row)(col).io.flush := io.flush
      cells(row)(col).io.stall := io.stall
    }
  }

}

class SACell(val width: Int = 8, val resDWidth: Int = 32) extends Module {
  val io = IO(new Bundle {
    val flush = Input(Bool())
    val stall = Input(Bool())
    val inHorizontal = Input(SInt(width.W))
    val outHorizontal = Output(SInt((width).W))
    val inVertical = Input(SInt(width.W))
    val outVertical = Output(SInt((width).W))
    val out = Output(SInt((resDWidth).W))
  })

  val regH = RegInit(0.S(width.W))
  val regV = RegInit(0.S(width.W))
  val res = RegInit(0.S(resDWidth.W))

  when(io.stall === false.B)
  {
    regH := io.inHorizontal
    regV := io.inVertical
    res := res + (io.inHorizontal * io.inVertical)
  }

  io.outHorizontal := regH
  io.outVertical := regV  
  io.out := res

  when(io.flush === true.B) {
    res := 0.S
    regH := 0.S
    regV := 0.S
  }
}


class MatAddressConvertor(cfg: MatMulConfig) extends Module {
  val io = IO(new Bundle {
    val elem_ptr = Input(UInt(log2Ceil(cfg.gemm_elem_num).W))
    val tile_x = Input(UInt(log2Ceil(cfg.gemm_matsize / cfg.matSize).W))
    val tile_y = Input(UInt(log2Ceil(cfg.gemm_matsize / cfg.matSize).W))
    val buf_sel = Input(UInt(log2Ceil(3).W))
    val addr = Output(UInt(log2Ceil(cfg.memory_size).W))
  })
  val offset_addr = Wire(UInt(log2Ceil(cfg.gemm_elem_num).W))
  offset_addr := ((io.tile_x * cfg.matSize.U + io.elem_ptr / cfg.matSize.U) * cfg.gemm_matsize.U
    + (io.tile_y * cfg.matSize.U + io.elem_ptr % cfg.matSize.U)) // 这里用移位操作更好
  io.addr := MuxCase(
    0.U,
    Seq(
      (io.buf_sel === cfg.SEL_A) -> cfg.matA_baseAddr.U,
      (io.buf_sel === cfg.SEL_B) -> cfg.matB_baseAddr.U,
      (io.buf_sel === cfg.SEL_C) -> cfg.matC_baseAddr.U
    )
  ) + offset_addr
}

