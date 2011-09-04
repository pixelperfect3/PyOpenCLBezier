__kernel void bezier(int degree, float detail, 
					 __global const float4 *a, __global float4 *c)
					 //__local float As[16])
{
	int gid = get_global_id(0)/9; // row number?
	// assume stride = (degree)
	// tensor-product patch of equal degrees
	int aid = gid * ((degree+1));
	
	// get all the points - quadratic curve
	float4 point1 = a[aid];
	float4 point2 = a[aid+1];
	float4 point3 = a[aid+2];
	
	// 1 - u
	float detail2 = 1.0 - detail;
	
	// do calculations
	float4 val = detail2 * detail2 * point1 + 
				 2.0 * detail2 * detail * point2 + 
				 detail * detail * point3; 
	
	c[gid] = val;
}

// Linear interpolation between a pair of points by a u-value
float4 lerp(float4 val1, float4 val2, float u) {
	return (1.0 - u) * val1 + u * val2;
}

// The optimized version:
// 15th August 2011
//__kernel __attribute__
//(( reqd_work_group_size (9, 1, 1)))
__kernel void bezierEval(__global const float4 *controlPoints, __global float2 *uvValues,
                        __global float4 *output, __local float4 *local_points, __local float4 *local_points2) {
 
 	// get the global id and then just output that
 	int gid = get_global_id(0)/9;
 	
 	// get the uv values - PROBLEM: NOT READING UV VALUES
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
 	
 	// TEMPORARY TEST: Try to just put the lid in there.
 	/* local_points[lid] = (float4)(lid, lid, lid, lid);
 	local_points[lid].s0 = (float)lid;
 	local_points[lid].s1 = (float)lid;
 	local_points[lid].s2 = (float)lid;
 	local_points[lid].s3 = (float)lid;*/
 	
 	// synchronize in work-group
 	barrier(CLK_LOCAL_MEM_FENCE);
 	
 	// only do it for certain work-items
 	/*if (lid == 0) {
 			output[gid] =   local_points[0] + 
 					   local_points[1] + 
 					   local_points[2] + 
 					   local_points[3] + 
 					   local_points[4] + 
 					   local_points[5] + 
 					   local_points[6] + 
 					   local_points[7] + 
 					   local_points[8]; 
 	}*/
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
 		
 		// only do it for the first work-item
 		if (lid == 0) {
 			float4 val13 = lerp(local_points2[0], local_points2[1], uv.x);
 			float4 val23 = lerp(local_points2[2], local_points2[3], uv.x);
 			
 			float4 newPoint3 = lerp(val13, val23, uv.y);
 			output[gid] = newPoint3;//(float4)(uv.s0, uv.s1, 0.0, 0.0);//newPoint3;
 		}
 	} 
 	
 	//output[gid] = b00;//controlPoints[gid];
 	
 	/* output[gid].s0 = gid;
 	output[gid].s1 = gid;
 	output[gid].s2 = gid;
 	output[gid].s3 = gid; */
 }