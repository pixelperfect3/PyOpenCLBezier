###################
## This is a simple PyOpenCL example
##
## The purpose is to evaluate Bicubic Bezier surfaces on the GPU, 
## AUTHOR: Shayan Javed
##

import sys
import pyopencl as cl
from numpy import *
import numpy

import datetime
import time

# Global variables
degreeU = 0
degreeV = 0
detail = 0.0

# the function to read a (modified) bezier view file - returns list of vertices
def readBezierFile(fileName):
    # open the file first
    file = open(fileName)
    
    # read all the text
    #str = file.read()
    
    # split it up line-by-line
    #lines = str.split('\n')
    
    # vertices
    vertices = []
    
    # keep reading until you reach end of file
    while True:
        # read 1 line
        line1 = file.readline()
        #print "Line: ", line1
        
        # EOF check
        if (len(line1) == 0):
            break
        
        # read group and number
        l1 = file.readline()
        l2 = file.readline()
        
        # split based on space and read the degrees
        line2 = l2.split()
        degree = int(line2[0])
        order = degree + 1
        
        # create numpy array
        # numpy array - make sure to specify astype as float32 to avoid /128 errors 
        #npVertices = empty( (order * order, 4)).astype(numpy.float32)
        
        index = 0
    
        while index < 16:
            # read line
            line = file.readline().split()
       
            # create a new list
            vertices.append(line[0])
            vertices.append(line[1])
            vertices.append(line[2])
            vertices.append(0)
            
            # increment
            index = index + 1
        
        ## END OF TRUE LOOP
        
    return vertices

# reads a file and returns all the text as a string
def readFile(fileName):
    # open the file first
    file = open(fileName, 'r')
    
    # read all the text
    str = "".join(file.readlines())
    
    return str

# the main function
def main(argv=None):
    if argv is None:
        argv = sys.argv
    
    # try to read from a file here - returns the array
    vertices = readBezierFile("cube2.bv")# * 10.0
    
    
    # create numpy array
    npVertices = array( vertices, dtype = float32)
    
    print npVertices
    
    print "Vertices:" 
    print npVertices.size
    
    # Number of patches (16 control points per patch)
    numPatches = (npVertices.size/16)/4
    print "NumPatches: ", numPatches
    
    # read the kernel file
    kernelString = readFile("bezier.cl")
    
    # the bezier curve TODO: fix to surface instead
    #curve1 = BezierCurve(len(vertices), 0.1, vertices)
    
    # Array of UV values - 36 in total (detail by 0.2)
    uvValues = empty( (36, 2)).astype(numpy.float32)
    
    index = 0
    for u in range(0, 12, 2): # step = 2
        for v in range(0, 12, 2):
            # conver the ints to floats
            fU = float(u)
            fV = float(v)
            uvValues[index] = [ fU/10.0, fV/10.0]
            index = index + 1
         

    print "\nUV Values: "
    print uvValues
    print uvValues.size
   
    ##########################
    ### OPENCL SETUP
    #########################
   
   
    # Platform test
    for found_platform in cl.get_platforms():
        if found_platform.name == 'NVIDIA CUDA':
            my_platform = found_platform
            print "Selected platform:", my_platform.name
   
    for device in my_platform.get_devices():
      dev_type = cl.device_type.to_string(device.type)
      if dev_type == 'GPU':
            dev = device
            print "Selected device: ", dev_type
   
    # context 
    # 2 ways of creating it. First directly specifying the dev_type and the 2nd just does it some way I guess
    ctx = cl.Context([dev])
    #ctx = cl.create_some_context()
    
    # command queue - NEW: ENABLE PROFILING
    cq = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    # memory flags
    mf = cl.mem_flags
    
    # create the Buffers
    
    # input buffers
    # the control point vertices
    vertex_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=npVertices)
    
    # the uv buffer
    uv_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=uvValues)
    
    # final output
    output_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, uvValues.nbytes * 2 * numPatches) # double the amount of value
    
    # the OpenCL program
    prg = cl.Program(ctx, kernelString).build()
    
    
    
    ##########################
    # EVALUATION STARTS HERE
    #########################
    
    # the global size (number of uv-values * number of patches): 36 * numPatches
    numUVs = uvValues.size/2 # 72/2 = 36
    globalSize = numUVs * numPatches

    localSize = numUVs # 36 threads per work-group?
    
    # timer start
    start = time.time()
    
    # evaluate
    exec_evt = prg.bezierEval2Multiple(cq, (globalSize,), (localSize,), vertex_buffer, uv_buffer, output_buffer) # cl.LocalMemory(9 * numpy.dtype('float32').itemsize * 4), cl.LocalMemory(4 * numpy.dtype('float32').itemsize * 4)) # local_buffer)#numpy.int32(degreeU), numpy.float32(u), vertex_buffer, inter_buf)
    exec_evt.wait()
    
    # find time taken
    # print exec_evt.profile.start
    elapsed = exec_evt.profile.end - exec_evt.profile.start
    print("Execution time of test: %g " % elapsed)
    
    # read back the result
    eval = empty( numUVs*4*numPatches).astype(numpy.float32)
    cl.enqueue_read_buffer(cq, output_buffer, eval).wait()
    
        
    print "It took: ", time.time() - start,"seconds."
    
    # Output to a file
    f = open('output2-bb', 'w')
    index = 0
    j = 6
    '''while index < numUVs * 4 * numPatches:
        val1 = eval[index]
        f.write(repr(round(val1, 2)).rjust(j))
        f.write("\n")
        index = index + 1'''
        
    while index < numUVs * numPatches:
        val1 = eval[index * 4]
        val2 = eval[index * 4 + 1]
        val3 = eval[index * 4 + 2]
        val4 = eval[index * 4 + 3]
        f.write(repr(round(val1, 2)).rjust(j))
        f.write(" ")
        f.write(repr(round(val2, 2)).rjust(j))
        f.write(" ")
        f.write(repr(round(val3, 2)).rjust(j))
        f.write(" ")
        f.write(repr(round(val4, 2)).rjust(j))
        f.write("\n")
        
        index = index + 1
        
        
    
    #print eval
    
# main invocation
if __name__ == "__main__":
    sys.exit(main())