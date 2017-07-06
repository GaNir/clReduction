
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <iostream>
#include <assert.h>
#include <png.h>
#include <unistd.h>
#include <CL/cl.hpp>
#include <sys/time.h>

#define BUFFER_SIZE (100*1024*1024)
#define NUM_THREADS 1024
#define NUM_ITERATIONS 10


// Auxiliary macros
#define Q(x) #x
#define QUOTE(x) Q(x)
#define ALIGN(X,A) (((X+(A-1))/A)*A)

cl_int getInfo(cl_program program, cl_device_id deviceId, cl_program_build_info infoEnum){
    cl_int res;
    cl_build_status build_status;
    char   *build_log;
    size_t retSize;

    clGetProgramBuildInfo(program, deviceId, infoEnum, 0, NULL, &retSize);
    build_log = new char[retSize+1];
    res = clGetProgramBuildInfo(program, deviceId, infoEnum, retSize, build_log, NULL);
    build_log[retSize] = '\0';
    std::cout << "============================== INFO ("<<infoEnum<<") ============"<< std::endl;
    if (infoEnum==CL_PROGRAM_BUILD_LOG) {
        std::cout << "======= Build log ========\n";
    }
    std::cout << build_log << std::endl;
    std::cout << "=================================================="<< std::endl;
    delete[] build_log;
    return res;
}


int main(int argc, char** argv){
    cl_int           res;
    cl_uint          retNum;
    cl_platform_id   platformId;
    cl_device_id     deviceId;
    cl_command_queue clQueue;
    cl_context       clContext;
    cl_program       clProgramImage;
    cl_kernel        clKernelImage;
    cl_mem           clBuff, clRes;
    cl_event         event;
    const char       PNG_FILE_NAME[] = "data/flower.png";
    void             *pBuffOut;
    void             *pBuffIn;
    cl_ulong         time_start, time_end, time_submit, time_queue;
    timespec         timeStart, timeEnd;
    const long       zeroInit = 0;
    double           avgTime  = 0;
    double           timeMs,gbps,avgGbps;
    long             resOfKernel = 0;
    long             checkSum = 0;


    std::cout<<"NUM_THREADS: " QUOTE(NUM_THREADS) << "\n";
    int *pBuff = (int*)calloc(BUFFER_SIZE,sizeof(int));


    // Data initialization
    for (int i = 0; i< BUFFER_SIZE; i++){
        pBuff[i] = (i%3);
    }

    // Arbitrary numbers to break data initialization pattern
    pBuff[0] = 10;
    pBuff[3] = 27;

    //// Alternative data initialization for debug
    // for (int i = 0; i< 1024; i++){
    //     pBuff[i] = 1;
    // }

    
    // ======= CL Bring up ===================================
    res = clGetPlatformIDs (1, &platformId, &retNum);
    assert(res==CL_SUCCESS);
    assert(retNum ==1);

    res = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_ALL, 1/*cl_uint num_entries*/,
                         &deviceId, &retNum/*cl_uint *num_devices*/);
    assert(res==CL_SUCCESS);
    assert(retNum ==1);
    
    const cl_context_properties properties[] = {CL_CONTEXT_PLATFORM,(long int)platformId,0};
    clContext = clCreateContext (properties,
            1,//cl_uint num_devices,
            &deviceId,
            NULL,//void ( CL_CALLBACK *pfn_notify
            NULL,//void *user_data,
            &res);//cl_int *errcode_ret
    assert(res==CL_SUCCESS);
 
    clQueue = clCreateCommandQueue (clContext, deviceId, CL_QUEUE_PROFILING_ENABLE/* properties*/, &res);
    assert(res==CL_SUCCESS);
    // ======= CL Bring up end =====================

    const char *kernelCodeImage[] ={
    "   #pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable\n"
    "   void kernel reductionKernel(global const int* buffer, global long* out){\n"
    "   const int idx = get_local_id(0);\n"
    "   local int shared["  QUOTE(NUM_THREADS) "];\n"
    "   shared[get_local_id(0)] = \n"
    "       buffer[4*get_group_id(0)*get_local_size(0) + get_local_id(0)] + \n"
    "       buffer[4*get_group_id(0)*get_local_size(0) + 1*get_local_size(0) + get_local_id(0)] +\n" 
    "       buffer[4*get_group_id(0)*get_local_size(0) + 2*get_local_size(0) + get_local_id(0)] +\n" 
    "       buffer[4*get_group_id(0)*get_local_size(0) + 3*get_local_size(0) + get_local_id(0)];\n" 
    "   barrier(CLK_LOCAL_MEM_FENCE);\n"  
    "   if (" QUOTE(NUM_THREADS) ">= 1024) { if (idx < 512) {shared[idx] += shared[idx+512];barrier(CLK_LOCAL_MEM_FENCE); } }\n"
    "   if (" QUOTE(NUM_THREADS) ">= 512 ) { if (idx < 256) {shared[idx] += shared[idx+256];barrier(CLK_LOCAL_MEM_FENCE); } }\n"
    "   if (" QUOTE(NUM_THREADS) ">= 256 ) { if (idx < 128) {shared[idx] += shared[idx+128];barrier(CLK_LOCAL_MEM_FENCE); } }\n"
    "   if (" QUOTE(NUM_THREADS) ">= 128 ) { if (idx < 64 ) {shared[idx] += shared[idx+64 ];barrier(CLK_LOCAL_MEM_FENCE); } }\n"
    "   if (idx < 32){\n"
    "       shared[idx] += shared[idx+32];\n"
    "       barrier(CLK_LOCAL_MEM_FENCE);\n"
    "       shared[idx] += shared[idx+16];\n"
    "       barrier(CLK_LOCAL_MEM_FENCE);\n"
    "       shared[idx] += shared[idx+8];\n"
    "       barrier(CLK_LOCAL_MEM_FENCE);\n"
    "       shared[idx] += shared[idx+4];\n"
    "       barrier(CLK_LOCAL_MEM_FENCE);\n"
    "       shared[idx] += shared[idx+2];\n"
    "       barrier(CLK_LOCAL_MEM_FENCE);\n"
    "       shared[idx] += shared[idx+1];\n"
 //   "       barrier(CLK_LOCAL_MEM_FENCE);\n"
    "   }\n"
    "   \n"
    "   \n"
    "   \n"
    "   if (idx == 0) atomic_add(out,shared[0]);\n"
    "   }\n" }; 


    clProgramImage = clCreateProgramWithSource ( clContext,
                                            1,// count,
                                            kernelCodeImage,//const char **strings,
                                            NULL,//const size_t *lengths,
                                            &res);

    assert(res == CL_SUCCESS);
    res =  clBuildProgram ( clProgramImage,
                            1,
                            &deviceId,
                            "-cl-nv-verbose",//const char *options,
                            NULL,//oid ( CL_CALLBACK *pfn_notify)(cl_program program,
                            NULL);//void *user_data)

    if (res == CL_BUILD_PROGRAM_FAILURE) {
     getInfo(clProgramImage, deviceId, CL_PROGRAM_BUILD_LOG);
     assert(0);
    }
    assert(res == CL_SUCCESS);
    
 
    clKernelImage = clCreateKernel ( clProgramImage,
                                "reductionKernel",
                                &res);
    assert(res == CL_SUCCESS);

 
    clBuff = clCreateBuffer( clContext,
                             CL_MEM_WRITE_ONLY,// cl_mem_flags flags,
                             BUFFER_SIZE*sizeof(int),  // size,
                             NULL, // void *host_ptr,
                             &res);// cl_int *errcode_ret)

    assert(res == CL_SUCCESS);
    assert(clBuff != NULL);

    res = clEnqueueWriteBuffer(clQueue, clBuff, CL_TRUE,
                                0, BUFFER_SIZE*sizeof(int), pBuff, 0, NULL, NULL);

    assert(res == CL_SUCCESS);
    
    clRes = clCreateBuffer(  clContext,
                             CL_MEM_READ_ONLY,// cl_mem_flags flags,
                             sizeof(long),  // size,
                             NULL, // void *host_ptr,
                             &res);// cl_int *errcode_ret)


    assert(res == CL_SUCCESS);
    assert(clRes != NULL);

    res = clSetKernelArg(clKernelImage,0,sizeof(cl_mem),&clBuff);
    assert(res == CL_SUCCESS);

    res = clSetKernelArg(clKernelImage,1,sizeof(cl_mem),&clRes);
    assert(res == CL_SUCCESS);

    size_t offset[1] = {0};  
    size_t local[1]  = {NUM_THREADS};
    size_t global[1] = {BUFFER_SIZE/4};//{ALIGN(buffSize,local[0])};


    for (int i=0;i<NUM_ITERATIONS;i++){

        res = clEnqueueWriteBuffer(clQueue, clRes, CL_TRUE,
                                    0, sizeof(long), &zeroInit, 0, NULL, NULL);

        res = clEnqueueNDRangeKernel(clQueue, clKernelImage, 1, offset, global, local, 0, NULL, &event);
        assert(res == CL_SUCCESS);
        clWaitForEvents(1, &event);
        clFinish(clQueue);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_start), &time_end, NULL);

        timeMs = (time_end-time_start)/1000000.0f;
        avgTime+=timeMs;
        gbps = ((BUFFER_SIZE/1024.0f/1024.0f/1024.0f*sizeof(int))/timeMs)*1000;
        avgGbps+=gbps;

        std::cout<<timeMs<< " ms    "<< gbps <<" GB/s"<<std::endl;
    }
    avgTime/=NUM_ITERATIONS;
    avgGbps/=NUM_ITERATIONS;

    std::cout<<"OpenCl average execution time: "<< avgTime << " ms    "<< avgGbps <<" GB/s\n\n";
    
    resOfKernel = 0;    
    res = clEnqueueReadBuffer(clQueue, clRes, CL_TRUE, 0, sizeof(long), &resOfKernel, 0, NULL, NULL);    
    assert(res == CL_SUCCESS);
    std::cout<<"Results: "<<resOfKernel;

    checkSum = 0;
    for (int i = 0; i < BUFFER_SIZE; i++){
        checkSum += pBuff[i];
    }

    if(checkSum != resOfKernel){
        std::cout<<" - expected "<<checkSum<<"\n";
        std::cout<<"======= ERROR ==========\n";
        assert(0);
    }
    
    std::cout<<" - ok\n\n";
  
    assert(res == CL_SUCCESS);
    res = clReleaseProgram(clProgramImage);
    assert(res == CL_SUCCESS);
    res = clReleaseMemObject(clBuff);
    assert(res == CL_SUCCESS);
    res = clReleaseMemObject(clRes);
    assert(res == CL_SUCCESS);
    res = clReleaseCommandQueue(clQueue);
    assert(res == CL_SUCCESS);
    res  = clReleaseContext(clContext);
    assert(res == CL_SUCCESS);
  
    free(pBuff);
 }