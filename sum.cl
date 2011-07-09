__kernel void bezier(int degree, float detail, 
					 __global const float4 *a, __global float4 *c)
					 //__local float As[16])
{
	int gid = get_global_id(0); // row number?
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