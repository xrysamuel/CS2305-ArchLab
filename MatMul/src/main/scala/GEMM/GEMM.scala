package GEMM

import chisel3._
import chisel3.util._

case class MatMulConfig(
    vecDWidth: Int = 8, // Precision of matrices A and B (8 bits)
    resDWidth: Int = 32, // Precision of result matrix C (32 bits)
    matSize: Int = 4, // Size of the matrices
    gemm_matsize: Int = 16, // Size of the matrices for GEMM
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

    // actual matrix size
    val N = Input(UInt(log2Ceil(cfg.gemm_matsize+1).W))
    val M = Input(UInt(log2Ceil(cfg.gemm_matsize+1).W))
    val K = Input(UInt(log2Ceil(cfg.gemm_matsize+1).W))
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

  mat_module.io.iMax := io.N - 1.U
  mat_module.io.jMax := io.M - 1.U
  mat_module.io.kMax := io.K - 1.U
}

class MatMem(MEM_ROW: Int, MEM_WIDTH: Int) extends Module {
  val io = IO(new Bundle {
    val addr = Input(UInt(log2Ceil(MEM_ROW).W))
    val writeEn = Input(Bool())
    val en = Input(Bool())
    val dataIn = Input(SInt(MEM_WIDTH.W))
    val dataOut = Output(SInt(MEM_WIDTH.W))
  })

  val mem = SyncReadMem(MEM_ROW,UInt(MEM_WIDTH.W))

  // single port
  when(io.writeEn & io.en) {
    mem.write(io.addr,io.dataIn.asUInt)
  }
  io.dataOut := mem.read(io.addr, io.en&&(!io.writeEn)).asSInt
}

// **********************************************************************

class MatMulModule(cfg: MatMulConfig) extends Module {
  val io = IO(new Bundle {
    val addr = Output(UInt(log2Ceil(cfg.memory_size).W))
    val writeEn = Output(Bool())
    val enable = Output(Bool())
    val dataIn = Input(SInt(cfg.memory_width.W))
    val dataOut = Output(SInt(cfg.memory_width.W))
    
    val iMax = Input(UInt(log2Ceil(cfg.gemm_matsize).W))
    val jMax = Input(UInt(log2Ceil(cfg.gemm_matsize).W))
    val kMax = Input(UInt(log2Ceil(cfg.gemm_matsize).W))

    val start = Input(Bool())
    val busy = Output(Bool())
  })

  val tileMulModule = Module(new TileMulModule(cfg))
  val iStart = Reg(UInt(log2Ceil(cfg.gemm_matsize + cfg.matSize).W))
  val kStart = Reg(UInt(log2Ceil(cfg.gemm_matsize + cfg.matSize).W))
  val iStartNext = Wire(UInt(log2Ceil(cfg.gemm_matsize + cfg.matSize).W))
  val kStartNext = Wire(UInt(log2Ceil(cfg.gemm_matsize + cfg.matSize).W))
  object State extends ChiselEnum {
    val idle, computing = Value
  }
  import State._
  val state = RegInit(idle)

  io.addr := tileMulModule.io.addr
  io.writeEn := tileMulModule.io.writeEn
  io.enable := tileMulModule.io.enable
  tileMulModule.io.dataIn := io.dataIn
  io.dataOut := tileMulModule.io.dataOut
  tileMulModule.io.iStart := iStart
  tileMulModule.io.kStart := kStart
  tileMulModule.io.iMask := Mux(io.iMax - iStart < cfg.matSize.U, io.iMax - iStart, (cfg.matSize - 1).U) 
  tileMulModule.io.kMask := Mux(io.kMax - kStart < cfg.matSize.U, io.kMax - kStart, (cfg.matSize - 1).U) 
  tileMulModule.io.jMax := io.jMax
  iStartNext := iStart + cfg.matSize.U
  kStartNext := kStart + cfg.matSize.U
  io.busy := false.B
  tileMulModule.io.start := false.B

  switch (state) {
    is (idle) {
      io.busy := false.B

      when(io.start) {
        tileMulModule.io.start := true.B
        iStart := 0.U
        kStart := 0.U
        
        state := computing
      }
    }

    is (computing) {
      io.busy := true.B
      tileMulModule.io.start := false.B

      when (tileMulModule.io.busy) {
        state := computing
      } .otherwise {
        when (kStartNext > io.kMax) {
          when (iStartNext > io.iMax) {
            state := idle
          } .otherwise {
            kStart := 0.U
            iStart := iStartNext
            tileMulModule.io.start := true.B
            state := computing
          }
        } .otherwise {
          kStart := kStartNext
          tileMulModule.io.start := true.B
          state := computing
        }
      }
    }
  }

  


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
    val busy = Output(Bool())
  })
  /*
   * TileMulModule: C[tile_i][tile_k] = The sum of A[tile_i][tile_j] * B[tile_j][tile_k] over tile_j
   * SA: C[tile_i][tile_k] += A[tile_i][tile_j] * B[tile_j][tile_k]
   * 
   * idle:  if start
   *            jump loop1 (flush)
   *        else 
   *            jump idle (flush)
   * loop1: counter = 0 (stall, busy)
   * loop2: aIn[:] = 0, bIn[:] = 0, index = max(counter - jMax, 0)
   *        dataIn = A[iStart+index][counter-index], jump loadB (busy)
   * loadA: dataIn = A[iStart+index][counter-index], bIn[index-1] = dataIn (stall, busy)
   * loadB: dataIn = B[counter-index][kStart+index], aIn[index] = dataIn (stall, busy)
   *        if index = min(counter, matSize - 1)
   *            if counter == matSize - 1 + jMax
   *                jump end1(stall, busy)
   *            else
   *                counter++, jump end2 (stall, busy)
   *        else
   *            index++, jump loadA (stall, busy)
   * end2:  bIn[index] = dataIn, jump loop2 (stall, busy)
   * end1:  bIn[index] = dataIn, counter = 0, jump storeC (stall, busy)
   * storeC:C[iStart+counter/matSize][kStart+counter%matSize] = out[counter/matSize][counter%matSize]
   *        aIn[:] = 0, bIn[:] = 0, 
   *        if counter == matSize * matSize - 1
   *            jump idle (busy)
   *        else
   *            counter++, jump storeC (busy)
   */

  val sa = Module(new SA(cfg.matSize, cfg.vecDWidth, cfg.resDWidth))
  val addressConvertor = Module(new AddressConvertor(cfg))
  val aIn = RegInit(VecInit(Seq.fill(cfg.matSize)(0.S(cfg.vecDWidth.W))))
  val bIn = RegInit(VecInit(Seq.fill(cfg.matSize)(0.S(cfg.vecDWidth.W))))
  val resetIn = VecInit(Seq.fill(cfg.matSize)(0.S(cfg.vecDWidth.W)))
  val idle :: loop1 :: loop2 :: loadA :: loadB :: end2 :: end1 :: storeC :: Nil = Enum(8)
  val state = RegInit(idle)
  val counter = RegInit(0.U(log2Ceil(cfg.matSize * cfg.matSize + cfg.gemm_matsize).W))
  val index = RegInit(0.U(log2Ceil(cfg.matSize).W))
  val indexMin = Wire(UInt(log2Ceil(cfg.matSize).W))
  val indexMax = Wire(UInt(log2Ceil(cfg.matSize).W))

  // default value
  sa.io.stall := false.B
  sa.io.flush := false.B
  io.busy := false.B
  for (i <- 0 until cfg.matSize) {
    sa.io.inHorizontal(i) := Mux(io.iMask < i.U, 0.S, aIn(i))
    sa.io.inVertical(i) := Mux(io.kMask < i.U, 0.S, bIn(i))
  }
  io.addr := addressConvertor.io.addr
  io.writeEn := false.B
  io.enable := false.B
  io.dataOut := 0.S
  addressConvertor.io.x := 0.U
  addressConvertor.io.y := 0.U
  addressConvertor.io.buf_sel := cfg.SEL_A
  indexMin := Mux(counter > io.jMax, counter - io.jMax, 0.U)
  indexMax := Mux(counter < (cfg.matSize - 1).U, counter, (cfg.matSize - 1).U)


  val readMemory = (sel: UInt, x: UInt, y: UInt) => {
    io.writeEn := false.B
    io.enable := true.B
    addressConvertor.io.x := x
    addressConvertor.io.y := y
    addressConvertor.io.buf_sel := sel
  }
  
  val writeMemory = (sel: UInt, x: UInt, y: UInt, dataOut: SInt) => {
    io.writeEn := true.B
    io.enable := true.B
    io.dataOut := dataOut
    addressConvertor.io.x := x
    addressConvertor.io.y := y
    addressConvertor.io.buf_sel := sel
  }

  val clearSAIn = () => {
    aIn := resetIn
    bIn := resetIn
  }

  switch (state) {
    is(idle) {
      sa.io.flush := true.B

      when(io.start) {
        state := loop1
      } .otherwise {
        state := idle
      }
    }

    is(loop1) {
      sa.io.stall := true.B
      io.busy := true.B
      
      counter := 0.U
      state := loop2
    }

    is(loop2) {
      io.busy := true.B

      clearSAIn()
      index := indexMin
      readMemory(cfg.SEL_A, io.iStart + indexMin, counter - indexMin)
      state := loadB
    }

    is(loadA) {
      sa.io.stall := true.B
      io.busy := true.B

      readMemory(cfg.SEL_A, io.iStart + index, counter - index)
      bIn(index - 1.U) := io.dataIn
      state := loadB
    }

    is(loadB) {
      sa.io.stall := true.B
      io.busy := true.B

      readMemory(cfg.SEL_B, counter - index, io.kStart + index)
      aIn(index) := io.dataIn
      when(index === indexMax) {
        when(counter - io.jMax === (cfg.matSize - 1).U) {
          state := end1
        } .otherwise {
          counter := counter + 1.U
          state := end2
        }
      } .otherwise {
        index := index + 1.U
        state := loadA
      }
    }

    is(end2) {
      sa.io.stall := true.B
      io.busy := true.B

      bIn(index) := io.dataIn
      state := loop2
    }

    is(end1) {
      sa.io.stall := true.B
      io.busy := true.B

      bIn(index) := io.dataIn
      counter := 0.U
      state := storeC
    }

    is(storeC) {
      io.busy := true.B

      writeMemory(cfg.SEL_C,
            io.iStart + counter / cfg.matSize.U,
            io.kStart + counter % cfg.matSize.U,
            sa.io.out(counter))
      clearSAIn()
      when(counter === (cfg.matSize * cfg.matSize - 1).U) {
        state := 0.U
      } .otherwise {
        counter := counter + 1.U
        state := storeC
      }
    }
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

  when(io.stall === false.B && io.flush === false.B)
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


class AddressConvertor(cfg: MatMulConfig) extends Module {
  val io = IO(new Bundle {
    val x = Input(UInt(log2Ceil(cfg.gemm_matsize).W))
    val y = Input(UInt(log2Ceil(cfg.gemm_matsize).W))
    val buf_sel = Input(UInt(log2Ceil(3).W))
    val addr = Output(UInt(log2Ceil(cfg.memory_size).W))
  })
  val offset_addr = Wire(UInt(log2Ceil(cfg.gemm_elem_num).W))
  offset_addr := io.x * cfg.gemm_matsize.U + io.y
  io.addr := MuxCase(
    0.U,
    Seq(
      (io.buf_sel === cfg.SEL_A) -> cfg.matA_baseAddr.U,
      (io.buf_sel === cfg.SEL_B) -> cfg.matB_baseAddr.U,
      (io.buf_sel === cfg.SEL_C) -> cfg.matC_baseAddr.U
    )
  ) + offset_addr
}

