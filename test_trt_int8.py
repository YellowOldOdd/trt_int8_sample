import numpy as np
import os
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import time

np_weight = np.random.random((768, 768)).astype(np.float32)
np_bias = np.random.random((1, 768)).astype(np.float32)

def convert_linear(trt_graph, in_tensor, name) :
  """
  Convert linear to trt.
  """
  weight = trt_graph.add_constant(shape=np_weight.shape,
                                  weights=np_weight)
  weight.name = name + "/kernel/data"

  op = trt_graph.add_matrix_multiply(in_tensor,
                                     trt.MatrixOperation.NONE,
                                     weight.get_output(0),
                                     trt.MatrixOperation.TRANSPOSE)
  op.name = name + "/kernel/matmul"
  
  bias = trt_graph.add_constant(shape=np_bias.shape, weights=np_bias)
  op.name = name + "/bias/data"
  op = trt_graph.add_elementwise(op.get_output(0),
                                 bias.get_output(0),
                                 trt.ElementWiseOperation.SUM)
  op.name = name + "/bias/add_bias"

  return op.get_output(0)

class BertInt8Calibrator(trt.IInt8EntropyCalibrator) :
  """
  Int8 Calibrator
  """
  def __init__(self) :
    trt.IInt8EntropyCalibrator.__init__(self) 
    fake_shape = [32, 768]

    self.cache_file = "bert_int8.calib"
    self.h_input = cuda.pagelocked_empty(trt.volume(trt.Dims(fake_shape)), dtype=np.float32)
    self.d_input = cuda.mem_alloc(self.h_input.nbytes)
    self.counter = 0
    
    fake_input = np.random.random(fake_shape).astype(np.float32)
    np.copyto(self.h_input, fake_input.ravel())
    cuda.memcpy_htod(self.d_input, self.h_input)

  def get_algorithm(self) :
    return trt.CalibrationAlgoType.ENTROPY_CALIBRATION

  def get_batch_size(self) :
    print('get_batch_size')
    return 1

  def write_calibration_cache(self, cache):
    print('write_calibration_cache')
    with open(self.cache_file, "wb") as f:
      f.write(cache)
  
  def read_calibration_cache(self):
    print('read_calibration_cache')
    # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
    if os.path.exists(self.cache_file):
      with open(self.cache_file, "rb") as f:
        return f.read()
    return None

  def get_batch(self, names):
    print('get_batch')
    print(names)
    if self.counter < 1 :
      self.counter = self.counter + 1 
      print("GET INT8 DATA {}.".format(self.counter))
      return [int(self.d_input)]
    else :
      return None

def gen_test_model(int8_mode) :
  """
  Build trt bert category model accroding to ptx code. 
  """
  trt_logger = trt.Logger(trt.Logger.INFO)
  trt_builder = trt.Builder(trt_logger)
  trt_graph = trt_builder.create_network()
  trt_builder.max_workspace_size = 1<<32
  trt_builder.max_batch_size = 100
  if int8_mode :
    trt_builder.int8_mode = True
    trt_builder.int8_calibrator = BertInt8Calibrator()
    trt_builder.strict_type_constraints = True
  
  test_input = trt_graph.add_input('test_input', 
                                   dtype=trt.float32,
                                   shape=(32, 768))
  
  trt_tensor = test_input
  for i in range(10) :
    trt_tensor = convert_linear(trt_graph, test_input, "test/loop{}".format(i))
    trt_tensor.name = "test/loop{}/output".format(i)

  trt_tensor.name = "test_output"
  trt_graph.mark_output(trt_tensor)
  trt_engine = trt_builder.build_cuda_engine(trt_graph)
  return trt_engine

def test_trt_int8() :
  fake_np_input = np.random.random((32, 768)).astype(np.float32)
  fake_np_output = np.zeros((32, 768)).astype(np.float32)
  
  # fp32
  # trt_engine = [gen_test_model(int8_mode = True), gen_test_model(int8_mode = False)]
  trt_engine = []
  print("=============================FP32=============================")
  trt_engine.append(gen_test_model(int8_mode = False))
  print("=============================INT8=============================")
  trt_engine.append(gen_test_model(int8_mode = True))

  for engine in trt_engine :
    with engine.create_execution_context() as context:
      h_data = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape('test_input')), 
                                     dtype=np.float32)
      d_data = cuda.mem_alloc(h_data.nbytes)
      
      out_h_data = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape('test_output')), 
                                         dtype=np.float32)
      out_d_data = cuda.mem_alloc(out_h_data.nbytes)
        
      np.copyto(h_data, fake_np_input.ravel())
      stream = cuda.Stream()
      cuda.memcpy_htod_async(d_data, h_data, stream)
      for i in range(10) :
        tstart = time.time()
        context.execute_async(bindings=[int(d_data), int(out_d_data)], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(out_h_data, out_d_data, stream)
        stream.synchronize()

        np.copyto(fake_np_output.ravel(), out_h_data)
        print("latency : {}".format(time.time() - tstart))
        print(fake_np_output)
      


test_trt_int8()