// Linear interpolation between a pair of points by a u-value
float4 lerp(float4 val1, float4 val2, float u) {
	return ((1.0 - u) * val1) + (u * val2);
}

// The supposedly optimized version:
// Sept. 3, 2011
__kernel void bezierEval(__global const float4 *controlPoints, __global float2 *uvValues,
                        __global float4 *output, __local float4 *local_points, __local float4 *local_points2) {
 
 	// get the global id
 	int gid = get_global_id(0)/9;
 	
 	// get the uv values - PROBLEM: NOT READING UV VALUES (FIXED)
 	float2 uv = uvValues[gid];//get_global_id(0)];
 	
 	// get the row
 	int lid = get_local_id(0);
 	int row = lid/3; 				// 3 = degree
 	
 	// get the 4 control points you'll need
 	/****************
 	  b10--b11
 	   |   |
 	  b00--b01
 	****************/
 	
 	float4 b00 = controlPoints[row + lid];
 	float4 b01 = controlPoints[row + lid+1];
 	float4 b10 = controlPoints[row + lid+4];
 	float4 b11 = controlPoints[row + lid+5];
 	
 	// do linear interpolation across u first
 	float4 val1 = lerp(b00, b01, uv.x);
 	float4 val2 = lerp(b10, b11, uv.x);
 	
 	// then across v
 	float4 newPoint = lerp(val1, val2, uv.y);
 	
 	// store it in the local memory
 	local_points[lid] = newPoint;
 	
 	// synchronize in work-group
 	barrier(CLK_LOCAL_MEM_FENCE);
 	
 	// only do it for certain work-items (now evaluating quadratic)
 	if (lid < 4) {
 		row = lid/2; 		// 2 = degree
 		float4 b002 = local_points[row + lid];
 		float4 b012 = local_points[row + lid+1];
 		float4 b102 = local_points[row + lid+3];
 		float4 b112 = local_points[row + lid+4];
 		
 		// do linear interpolation across u first
 		float4 val12 = lerp(b002, b012, uv.x);
 		float4 val22 = lerp(b102, b112, uv.x);
 	
 		// then across v
 		float4 newPoint2 = lerp(val12, val22, uv.y);
 		
 		local_points2[lid] = newPoint2;
 		barrier(CLK_LOCAL_MEM_FENCE);
 		
 		//output[gid] = newPoint2;
 		
 		// only do it for the first work-item
 		if (lid == 1) {
 			float4 val13 = lerp(local_points2[0], local_points2[1], uv.x);
 			float4 val23 = lerp(local_points2[2], local_points2[3], uv.x);
 			
 			float4 newPoint3 = lerp(val13, val23, uv.y);
 			int x = row + lid + 5;
 			//output[gid] = (float4)(x, 0, 0, 0);
 			output[gid] = newPoint3;
 		}
 	}
 	
 	int x = row + lid;
 	//output[gid] = newPoint;//(float4)(x, 0, 0.0, 0.0);
 }
 

//The 2nd supposedly optimized version (for multiple Bezier patches read from BezierView files):
//October 1, 2011
__kernel void bezierEvalMultiple(int numPatches, __global const float4 *controlPoints, __global float2 *uvValues,
                     __global float4 *output, __local float4 *local_points, __local float4 *local_points2) {

	// get the global id
	int gid = get_global_id(0)/get_local_size(0); // divide by 9
	
	// get the patch number
	int numPatch = gid/36; //get_local_size(0);
	
	// get the uv values 
	float2 uv = uvValues[gid];//get_global_id(0)];
	
	//output[gid] = (float4)(numPatch, numPatch, 0, 0);
	
	// get the row
	int lid = get_local_id(0);
	int patchNum = numPatch * 16; 	// the patch to deal with
	int row = lid/3;
	int index = row + numPatch + lid;
	
	// get the 4 control points you'll need
	/****************
	  b10--b11
	   |   |
	  b00--b01
	****************/
	
	float4 b00 = controlPoints[index];
	float4 b01 = controlPoints[index+1];
	float4 b10 = controlPoints[index+4];
	float4 b11 = controlPoints[index+5];
	
	// do linear interpolation across u first
	float4 val1 = lerp(b00, b01, uv.x);
	float4 val2 = lerp(b10, b11, uv.x);
	
	// then across v
	float4 newPoint = lerp(val1, val2, uv.y);
	
	// store it in the local memory
	local_points[lid] = newPoint;
	
	// synchronize in work-group
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// only do it for certain work-items (now evaluating quadratic)
	if (lid < 4) {
		row = lid/2; 		// 2 = degree
		float4 b002 = local_points[row + lid];
		float4 b012 = local_points[row + lid+1];
		float4 b102 = local_points[row + lid+3];
		float4 b112 = local_points[row + lid+4];
		
		// do linear interpolation across u first
		float4 val12 = lerp(b002, b012, uv.x);
		float4 val22 = lerp(b102, b112, uv.x);
	
		// then across v
		float4 newPoint2 = lerp(val12, val22, uv.y);
		
		local_points2[lid] = newPoint2;
		barrier(CLK_LOCAL_MEM_FENCE);
		
		//output[gid] = newPoint2;
		
		// only do it for the first work-item
		if (lid == 1) {
			float4 val13 = lerp(local_points2[0], local_points2[1], uv.x);
			float4 val23 = lerp(local_points2[2], local_points2[3], uv.x);
			
			float4 newPoint3 = lerp(val13, val23, uv.y);
			//output[gid] = (float4)(x, 0, 0, 0);
			output[gid] = newPoint3;
		}
	}
	
	//output[gid] = newPoint;//(float4)(x, 0, 0.0, 0.0); */
}

// function for evaluating cubic bezier curves
float4 evalCubicCurve(float4 b00, float4 b01, float4 b02, float4 b03, float u) {
 	return (1-u) * (1-u) * (1-u) *     b00 + 
			3    * (1-u) * (1-u) * u * b01 + 
			3    * (1-u) *    u  * u * b02 + 
			u    *     u *    u  *     b03;
}
 
 // The 2nd method of evaluating - FOR MULTIPLE PATCHES
 // Evaluates 1 UV-value pair per thread (work-item)
__kernel void bezierEval2Multiple(__global const float4 *controlPoints, __global float2 *uvValues,
                        __global float4 *output) {
 
 	// get the global id and the corresponding patch number
 	int gid = get_global_id(0);
 	
	// get the patch number - the patch info we will have to read
	int numPatch = gid/get_local_size(0);
	
 	// get the uv values - local id goes from 0-35
	int lid = get_local_id(0);
 	float2 uv = uvValues[lid];//uvValues[gid];
 	
	// evaluate row1
	float4 b00 = evalCubicCurve(controlPoints[numPatch * 16 + 0], controlPoints[numPatch * 16 + 1], controlPoints[numPatch * 16 + 2], controlPoints[numPatch * 16 + 3], uv.x);
	float4 b01 = evalCubicCurve(controlPoints[numPatch * 16 + 4], controlPoints[numPatch * 16 + 5], controlPoints[numPatch * 16 + 6], controlPoints[numPatch * 16 + 7], uv.x);
	float4 b02 = evalCubicCurve(controlPoints[numPatch * 16 + 8], controlPoints[numPatch * 16 + 9], controlPoints[numPatch * 16 + 10], controlPoints[numPatch * 16 + 11], uv.x);
	float4 b03 = evalCubicCurve(controlPoints[numPatch * 16 + 12], controlPoints[numPatch * 16 + 13], controlPoints[numPatch * 16 + 14], controlPoints[numPatch * 16 + 15], uv.x);
 		
	// evaluated point
 	output[gid] = evalCubicCurve(b00, b01, b02, b03, uv.y);
}
 
 // The 2nd method of evaluating 
 // Evaluates 1 UV-value pair per thread (work-item)
__kernel void bezierEval2(__global const float4 *controlPoints, __global float2 *uvValues,
                        __global float4 *output) {
 
 	// get the global id
 	int gid = get_global_id(0);
 	
 	// get the uv values
 	float2 uv = uvValues[gid];
 	
	// evaluate row1
	float4 b00 = evalCubicCurve(controlPoints[0], controlPoints[1], controlPoints[2], controlPoints[3], uv.x);
	float4 b01 = evalCubicCurve(controlPoints[4], controlPoints[5], controlPoints[6], controlPoints[7], uv.x);
	float4 b02 = evalCubicCurve(controlPoints[8], controlPoints[9], controlPoints[10], controlPoints[11], uv.x);
	float4 b03 = evalCubicCurve(controlPoints[12], controlPoints[13], controlPoints[14], controlPoints[15], uv.x);
 		
	// evaluated point
 	output[gid] = evalCubicCurve(b00, b01, b02, b03, uv.y);
}