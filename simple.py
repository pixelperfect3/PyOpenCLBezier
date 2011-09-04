###################
## This is a simple PyOpenCL example
##
## The purpose is to evaluate Bezier Curves on the GPU, 
## AUTHOR: Shayan Javed
##

import sys
import pyopencl as cl
from numpy import *
import numpy
import time

# Global variables
degreeU = 0
degreeV = 0
detail = 0.0


# class definition begins here
# degree = the degree of the bezier curve
# detail = how finely rendererd the curve should be (increments 
class BezierCurve:
    def __init__(self, degree=4, detail=0.1, vertices=[]):
        self.degree = degree
        self.detail = detail
        
        # vertices is a 2d-list [ [x0 y0 z0] [x1 y1 z1] ... [xN yN zN] ]
        self.vertices = vertices

# the function to read a bezier file - returns list of vertices
def readBezierFile(fileName):
    # open the file first
    file = open(fileName)
    
    # read all the text
    str = file.read()
    
    # split it up line-by-line
    lines = str.split('\n')
    
    # first line - degree + detail
    line1 = lines[0].split()
    global degreeU
    global degreeV
    global detail
    degreeU = int(line1[0])
    degreeV = int(line1[1])
    detail = float(line1[2])
    order = int(degreeU) + 1
    
    # create list
    verticesL = []#[[0] * 3] * order
    # numpy array - make sure to specify astype as float32 to avoid /128 errors 
    vertices = empty( (order * order, 4)).astype(numpy.float32)
    #vertices.dtype = float_
    
    # go through the rest of the lines, read in the curve
    index = 1
    
    #print len(lines)
    #print lines
    while index <= (len(lines) - 1):
        # read line
        line = lines[index].split()
       
        # create a new list
        newList = [line[0], line[1], line[2], 0]
        
         # assign values
        vertices[index-1] = newList
        
        # basic list
        #verticesL.append(float(line[0]) + 10.0)
        #verticesL.append(float(line[1]) * 10.0)
        #verticesL.append(float(line[2]) * 10.0)
        
        # increment
        index = index + 1
    
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
    vertices = readBezierFile("curve1") * 10.0
    print vertices
    
    # flatten the numpy array
    #vertices.shape = (4 * 3)
    #print vertices.shape
    
    print "Vertices:" 
    print vertices.size
    
    # read the kernel file
    kernelString = readFile("sum.cl")
    
    # the bezier curve TODO: fix to surface instead
    #curve1 = BezierCurve(len(vertices), 0.1, vertices)
    
    # Array of UV values - 36 in total (detail by 0.2)
    uvValues = empty( (36, 2)).astype(numpy.float32)
    print uvValues
    index = 0
    for u in range(0, 12, 2): # step = 2
        for v in range(0, 12, 2):
            # conver the ints to floats
            fU = float(u)
            fV = float(v)
            uvValues[index] = [ fU/10.0, fV/10.0]
            index = index + 1
         
    #uvValues = array([ [0.0, 0.0, 0.0, 0.0], [0.0, 0.2, 0.0, 0.0], [0.0, 0.4, 0.0, 0.0], [0.0, 0.6, 0.0, 0.0], [0.0, 0.8, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
    #             [0.2, 0.0, 0.0, 0.0], [0.2, 0.2, 0.0, 0.0], [0.2, 0.4, 0.0, 0.0], [0.2, 0.6, 0.0, 0.0], [0.2, 0.8, 0.0, 0.0], [0.2, 1.0, 0.0, 0.0],
    #             [0.4, 0.0, 0.0, 0.0], [0.4, 0.2, 0.0, 0.0], [0.4, 0.4, 0.0, 0.0], [0.4, 0.6, 0.0, 0.0], [0.4, 0.8, 0.0, 0.0], [0.4, 1.0, 0.0, 0.0],
    #             [0.6, 0.0, 0.0, 0.0], [0.6, 0.2, 0.0, 0.0], [0.6, 0.4, 0.0, 0.0], [0.6, 0.6, 0.0, 0.0], [0.6, 0.8, 0.0, 0.0], [0.6, 1.0, 0.0, 0.0],
    #             [0.8, 0.0, 0.0, 0.0], [0.8, 0.2, 0.0, 0.0], [0.8, 0.4, 0.0, 0.0], [0.8, 0.6, 0.0, 0.0], [0.8, 0.8, 0.0, 0.0], [0.8, 1.0, 0.0, 0.0],
    #             [1.0, 0.0, 0.0, 0.0], [1.0, 0.2, 0.0, 0.0], [1.0, 0.4, 0.0, 0.0], [1.0, 0.6, 0.0, 0.0], [1.0, 0.8, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0] ]);
    print "\nUV Values: "
    print uvValues
    print uvValues.size
   
    ##########################
    ### OPENCL SETUP
    #########################
   
    # context
    ctx = cl.create_some_context()
    # command queue
    cq = cl.CommandQueue(ctx)
    # memory flags
    mf = cl.mem_flags
    
    # create the Buffers
    
    # input buffers
    # the control point vertices
    vertex_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vertices)
    
    # the uv buffer
    uv_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=uvValues)
    
    # intermediate output buffer
    #inter_buf = cl.Buffer(ctx, mf.READ_WRITE, vertices.nbytes - 32)
    
    # create intermediate local buffer
    #local_buf
    
    # final output
    #output_buf = cl.Buffer(ctx, mf.WRITE_ONLY, 32) # just one value
    output_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, uvValues.nbytes * 2) # double the amount of value
    
    # the OpenCL program
    prg = cl.Program(ctx, kernelString).build()
    
    # timer start
    start = time.time()
    
    # the global size (number of uv-values): 36 work-items
    globalSize = 36 * 9

    localSize = 9 # for bicubic patches
    
    # evaluate
    prg.bezierEval(cq, (globalSize,), (localSize,), vertex_buffer, uv_buffer, output_buffer, cl.LocalMemory(9 * 9), cl.LocalMemory(9 * 4)) # local_buffer)#numpy.int32(degreeU), numpy.float32(u), vertex_buffer, inter_buf)
    
    # read back the result - works!
    #eval = empty((36, 4)).astype(numpy.float32);
    eval = empty( 36*4).astype(numpy.float32)#numpy.empty_like(vertices)
    cl.enqueue_read_buffer(cq, output_buffer, eval).wait()
    
    
    ####################
    #### NAIVE METHOD ########
    #### Loop through all possible u-values - naive method
    ''' result = []
    u = 0.0
    v = 0.0
    while u <= 1.0:
        # call the program
        # NOTE: Need to convert to proper formats using numpy
        prg.bezier(cq, ((degreeU+1),), None, numpy.int32(degreeU), numpy.float32(u), vertex_buffer, inter_buf)
        
        v = 0.0
        
        while v <= 1.0:
            # call again
            prg.bezier(cq, (1,), None, numpy.int32(degreeU), numpy.float32(v), inter_buf, output_buf)
            
            # get the result back
            eval = empty( 4).astype(numpy.float32)#numpy.empty_like(vertices)
            cl.enqueue_read_buffer(cq, output_buf, eval).wait()
            
            # add it to results
            result.append(eval)
            
            v += 0.5
        
        # increment
        u += 0.5 '''
        
    print "It took: ", time.time() - start,"seconds."
    print eval
    #print "\nResult:"
    #print result
    ########### END OF NAIVE METHOD ###############
    
# main invocation
if __name__ == "__main__":
    sys.exit(main())