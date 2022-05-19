/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Matrix transpose with Metal
* Device code.
*/

#define BLOCK_DIM 16
#define INC(x, l) min(x + 1, l - 1)

#define CONFIG_WIDTH            0 
#define CONFIG_HEIGHT           1 
#define CONFIG_X            	2 
#define CONFIG_Y           		3 
#define CONFIG_HALF_W           4
#define CONFIG_WH           	5
#define CONFIG_SPEC          	6

#define RECT_KX            		0
#define RECT_KY             	1
#define RECT_KX_2            	2
#define RECT_KY_2            	3
#define RECT_KX_KY             	4

/* This kernel is optimized to ensure all global reads and writes are coalesced,and to avoid bank conflicts in shared memory.  This kernel is up to 11x faster than the naive kernel below.  Note that the shared memory array is sized to(BLOCK_DIM+1)*BLOCK_DIM.  This pads each row of the 2D block in shared memory so that bank conflicts do not occur when threads address the array column-wise.*/


float4 dataUchar4ToFloat4(const uint uiPacked)
{
    float4 data_f4;
    data_f4.x =  uiPacked        & 0xff;
    data_f4.y = (uiPacked >>  8) & 0xff;
    data_f4.z = (uiPacked >> 16) & 0xff;
    data_f4.w = (uiPacked >> 24) & 0xff;
    return data_f4;
}

// Inline device function to convert floating point rgba color to 32-bit unsigned integer
//*****************************************************************

uint dataFloat4ToUchar4(const float4 data_f4)
{
    uint uiPacked = 0U;
    uiPacked |= 0x000000FF &   (uint)data_f4.x;
    uiPacked |= 0x0000FF00 & (((uint)data_f4.y) << 8);
    uiPacked |= 0x00FF0000 & (((uint)data_f4.z) << 16);
    uiPacked |= 0xFF000000 & (((uint)data_f4.w) << 24);
    return uiPacked;
}

/*****************************************************************************************************
//  function:
//      transpose the image from vertical to horizal or from horizal to vertical 
//  parameters:
//      idata    - pointer to input data (RGBA image packed into 32-bit floats)
//      odata    - pointer to output data 
//      width    - image width
//      height   - image height
//      block    - pointer to temp buffer(work-item)

//  revised 2019.09.07
*****************************************************************************************************/
kernel void Transpose
(
    device uchar* idata, 
             int    width, 
             int    height, 
    thread  uchar* block,
    device uchar* odata,
	uint2 index [[thread_position_in_grid]],
	uint2 localSize [[thread_execution_width]],
	uint2 groupId [[threadgroup_position_in_grid]],
	uint2 localId [[thread_position_in_threadgroup]]
)
{
    // read the matrix tile into shared memory
    int xIndex     = index.x;
    int yIndex     = index.y;

    int localSizeX = localSize.x;
    int localSizeY = localSize.y;

    int group_id_X = groupId.x;
    int group_id_Y = groupId.y;

    if ((xIndex < width) && (yIndex < height))
    {
        int index_inp    = mad24(yIndex, width, xIndex);
        int index_blk    = localId.y * (localSize.x+1) + localId.x;

        block[index_blk] = idata[index_inp];
    }

    //barrier(CLK_LOCAL_MEM_FENCE);
	threadgroup_barrier(mem_threadgroup)

    // write the transposed matrix tile to global memory
    xIndex = mul24(group_id_Y, localSizeY) + localId.x;
    yIndex = mul24(group_id_X, localSizeX) + localId.y;

    if ((xIndex < height) && (yIndex  < width))
    {
        int index_blk    = localId.x * (localSize.y+1) + localId.y;
        int index_out    = mad24(yIndex, height, xIndex) ;
        odata[index_out] = block[index_blk];
    }
}

// Recursive Gaussian filter 
//*****************************************************************
//  function:
//      blur the image from vertical with gaussian filter
//  parameters:
//      puiDataInp_ - pointer to  input data
//      puiDataOut_ - pointer to output data 
//      iWidth      - image width
//      iHeight     - image height
//      a0-a3, b1, b2, coefp, coefn - filter parameters
//
//*****************************************************************
kernel void GaussianBlur
(
    device const uint* puiDataInp_, 
                   int   iWidth,
                   int   iHeight, 
                   float a0,
                   float a1,
                   float a2,
                   float a3,
                   float b1,
                   float b2,
                   float coefp,
                   float coefn,
    device       uint* puiDataOut_
	uint2 groupId [[threadgroup_position_in_grid]]
	uint2 localSize [[thread_execution_width]]
	uint2 localId [[thread_position_in_threadgroup]]
)
{
    // compute X pixel location and check in-bounds
   
    int localSizeX = localSize.x;
    int localSizeY = localSize.y;
    int group_id_x = groupId.x;
    int local_id_x = localId.x;
    int nIdX = mad24(group_id_x, localSizeX, local_id_x);

     device uint *puiDataInp = puiDataInp_;
     device uint *puiDataOut = puiDataOut_;

    if (nIdX >= iWidth) return;

    // advance global pointers to correct column for this work item and x position
    puiDataInp += nIdX;
    puiDataOut += nIdX;

    // start forward filter pass
    float4 xp = (float4)0.0f;  // previous input
    float4 yp = (float4)0.0f;  // previous output
    float4 yb = (float4)0.0f;  // previous output by 2

    #ifdef YL_CL_CLAMP_TO_EDGE
    xp = dataUchar4ToFloat4(*puiDataInp);
    yb = xp * (float4)coefp;
    yp = yb;
    #endif

    for (int nIdY = 0; nIdY < iHeight; nIdY++)
    {
        float4 xc = dataUchar4ToFloat4(*puiDataInp);
        float4 yc = (xc * a0) + (xp * a1) - (yp * b1) - (yb * b2);
        *puiDataOut = dataFloat4ToUchar4(yc);
        xp = xc;
        yb = yp;
        yp = yc;
        puiDataInp += iWidth;    // move to next row
        puiDataOut += iWidth;    // move to next row
    }

    // reset global pointers to point to last element in column for this work item and x position
    puiDataInp -= iWidth;
    puiDataOut -= iWidth;

    // start reverse filter pass: ensures response is symmetrical
    float4 xn = (float4)0.0f;
    float4 xa = (float4)0.0f;
    float4 yn = (float4)0.0f;
    float4 ya = (float4)0.0f;

    #ifdef YL_CL_CLAMP_TO_EDGE
    xn = dataUchar4ToFloat4(*puiDataInp);
    xa = xn;
    yn = xn * (float4)coefn;
    ya = yn;
    #endif

    for (int nIdY = iHeight - 1; nIdY > -1; nIdY--)
    {
        float4 xc = dataUchar4ToFloat4(*puiDataInp);
        float4 yc = (xn * a2) + (xa * a3) - (yn * b1) - (ya * b2);
        xa = xn;
        xn = xc;
        ya = yn;
        yn = yc;
        float4 fTemp = dataUchar4ToFloat4(*puiDataOut) + yc;

        *puiDataOut = dataFloat4ToUchar4(fTemp);
        puiDataInp -= iWidth;  // move to previous row
        puiDataOut -= iWidth;  // move to previous row
    }
}

kernel void Fusion
(
    device const uint * puiDataIn,
    device const uchar* pucFrame ,
    device const int  * piLabelIn,
                   int    iWidth,
                   int    iHeight,
                   float  fMinThresh,
                   float  fMaxThresh,
    device       uint * puiDataOut,
	uint2 groupId [[threadgroup_position_in_grid]],
	uint2 localId [[thread_position_in_threadgroup]]
)
{
    float4 fl4_BlurValue = (float4)0.0f;
    float4 fl4_FusnValue = (float4)0.0f;
    uchar4 uc4_FramValue = (uchar4)0.0f;
    float4 fl4_ForeValue = (float4)0.0f;
    float4 fl4_BackValue = (float4)0.0f;
    float4 fl4_DivdScale = (float4)255.0f;
    float4 fl4_EqulScale = (float4)1.0f;
    float4 fl4_FramValue = (float4)0.0f;

    int nOffSet = 0;
    uint X = (groupId.x * localSize.x) + localId.x;
    uint Y = (groupId.y * localSize.y) + localId.y;

    if (X < iWidth && Y < iHeight)
    {
        nOffSet = Y * iWidth + X;
        fl4_BlurValue = dataUchar4ToFloat4(puiDataIn[nOffSet]);
        fl4_ForeValue = dataUchar4ToFloat4(piLabelIn[nOffSet]);

        uc4_FramValue = vload4(nOffSet, pucFrame);
        fl4_FusnValue = (float4)(fl4_BlurValue);
        fl4_ForeValue = fl4_ForeValue / fl4_DivdScale;
        fl4_BackValue = fl4_EqulScale - fl4_ForeValue;

        fl4_FramValue = (float4)(uc4_FramValue.x, uc4_FramValue.y, uc4_FramValue.z, uc4_FramValue.w);

        if ((fl4_ForeValue.x >= fMinThresh) && (fl4_ForeValue.x < fMaxThresh))
        {
            fl4_FusnValue.x = fl4_FramValue.x * fl4_ForeValue.x + fl4_FusnValue.x * fl4_BackValue.x;
        }
        else if (fl4_ForeValue.x >= fMaxThresh)
        {
            fl4_FusnValue.x = fl4_FramValue.x;
        }

        if ((fl4_ForeValue.y >= fMinThresh) && (fl4_ForeValue.y < fMaxThresh))
        {
            fl4_FusnValue.y = fl4_FramValue.y * fl4_ForeValue.y + fl4_FusnValue.y * fl4_BackValue.y;
        }
        else if (fl4_ForeValue.y >= fMaxThresh)
        {
            fl4_FusnValue.y = fl4_FramValue.y;
        }

        if ((fl4_ForeValue.z >= fMinThresh) && (fl4_ForeValue.z < fMaxThresh))
        {
            fl4_FusnValue.z = fl4_FramValue.z * fl4_ForeValue.z +  fl4_FusnValue.z * fl4_BackValue.z;
        }
        else if (fl4_ForeValue.z >= fMaxThresh)
        {
            fl4_FusnValue.z = fl4_FramValue.z;
        }

        if ((fl4_ForeValue.w >= fMinThresh) && (fl4_ForeValue.w < fMaxThresh))
        {
            fl4_FusnValue.w = fl4_FramValue.w * fl4_ForeValue.w +  fl4_FusnValue.w * fl4_BackValue.w;
        }
        else if (fl4_ForeValue.w >= fMaxThresh)
        {
            fl4_FusnValue.w = fl4_FramValue.w;
        }

        puiDataOut[nOffSet] = dataFloat4ToUchar4(fl4_FusnValue);
    }
}

//*****************************************************************
//  function:
//      Image resize 
//      
//  parameters:
//      srcptr      - pointer to input data (plannar image)
//      src_step    - input image step 
//      src_offset  - input image offset
//      src_cols    - input image width
//      src_rows    - input image height
//      ifx         - scale of x
//      ify         - scale of y
//      dstptr      - pointer to desti data (plannar image)
//      dst_step    - dst image step 
//      dst_offset  - dst image offset
//      dst_cols    - dst image width
//      dst_rows    - dst image height
//
//  revised 2019.10.09
//*****************************************************************
kernel void ResizeBillinear
(
    device const uchar * srcptr,
                   int     src_step,
                   int     src_rows,
                   int     src_cols,
                   float   ifx,
                   float   ify, 
    device       uchar * dstptr, 
                   int     dst_step, 
                   int     dst_rows,
                   int     dst_cols,
	uint2 index [[thread_position_in_grid]]
)
{
    int dx = index.x;
    int dy = index.y;

    int nOffSet = 0;
    uchar pixel;
    float inv_fx = 1.0 / ifx;
    float inv_fy = 1.0 / ify;

    if (dx < dst_cols && dy < dst_rows)
    {
        float sx = ((dx + 0.5f) * inv_fx - 0.5f);
        float sy = ((dy + 0.5f) * inv_fy - 0.5f);

        int x = floor(sx);
        int y = floor(sy);
        float u = sx - x;
        float v = sy - y;

        if (x < 0) 
        {
            x = 0;
            u = 0;
        }

        if (x >= src_cols)
        {
            x = src_cols - 1;
            u = 0;
        }

        if (y < 0)
        {
            y = 0;
            v = 0;
        }

        if (y >= src_rows)
        {
            y = src_rows - 1;
            v = 0;
        }

        int y_ = INC(y, src_rows);
        int x_ = INC(x, src_cols);
        float u1 = 1.f - u;
        float v1 = 1.f - v;

        uchar data0 = (srcptr[mad24(y , src_step, x )]);
        uchar data1 = (srcptr[mad24(y , src_step, x_)]);
        uchar data2 = (srcptr[mad24(y_, src_step, x )]);
        uchar data3 = (srcptr[mad24(y_, src_step, x_)]);

        float fval = u1 * v1 * data0 + u * v1 * data1 + u1 * v * data2 + u * v * data3;

        nOffSet = mad24(dy, dst_step, dx);
        fval    = (fval + 0.5) <=  0 ? 0   : (fval + 0.5);
        pixel   =  fval        > 255 ? 255 :  fval       ;
        dstptr[nOffSet] = pixel;
    }
}


//NV12转I420,并且输出resize之后的I420
kernel void ScaleConvert_NV12_I420(	
    device const uchar* src,	
	constant int* srcConfig,	
    device uchar* dst_y,
	device uchar* dst_u,
	device uchar* dst_v,
	constant int* dstConfig,
	constant float* RectConfig,
	uint2 index [[thread_position_in_grid]])	
{	
	int2 nowCoordinate;				 
    nowCoordinate.x = index.x;
    nowCoordinate.y = index.y;
	
	float2 srcCoordinate;

	srcCoordinate.x = srcConfig[CONFIG_X] + nowCoordinate.x * RectConfig[RECT_KX];
	srcCoordinate.y = srcConfig[CONFIG_Y] + nowCoordinate.y * RectConfig[RECT_KY];
	
	int2 dstCoordinate;	
	dstCoordinate.x = nowCoordinate.x + dstConfig[CONFIG_X];
	dstCoordinate.y = nowCoordinate.y + dstConfig[CONFIG_Y];
	
	float2 decimal;
	int2 srcIntCoordinate = convert_int2(srcCoordinate);
	decimal.x = srcCoordinate.x - srcIntCoordinate.x;
	decimal.y = srcCoordinate.y - srcIntCoordinate.y;	
	int srcNum = mad24(srcIntCoordinate.y, srcConfig[CONFIG_WIDTH], srcIntCoordinate.x);
	int dstNum = mad24(dstCoordinate.y, dstConfig[CONFIG_WIDTH], dstCoordinate.x); 

	//dst[dstNum] = src[srcNum]; 

	uchar2 x00_10;
	uchar2 x01_11;
	x00_10 = (*((device uchar2 *)&src[srcNum]));
	x01_11 = (*((device uchar2 *)&src[srcNum+srcConfig[CONFIG_WIDTH]]));
	x00_10.x = x00_10.x + (x01_11.x - x00_10.x) * decimal.y;
	x00_10.y = x00_10.y + (x01_11.y - x00_10.y) * decimal.y;
	x00_10.x = x00_10.x + (x00_10.y - x00_10.x) * decimal.x;
	dst_y[dstNum] = x00_10.x;
		
	if (((dstCoordinate.x % 2) == 0) && ((dstCoordinate.y % 2) == 0))
	{
	    srcIntCoordinate.x = srcIntCoordinate.x & 0xfffffffe;
		srcIntCoordinate.y = srcIntCoordinate.y & 0xfffffffe;
        
//		if(srcIntCoordinate.y > srcConfig[CONFIG_HEIGHT] || srcIntCoordinate.x > srcConfig[CONFIG_WIDTH])
//			return;

		int srcUVnum = mad24(srcConfig[CONFIG_HALF_W], srcIntCoordinate.y, srcIntCoordinate.x + srcConfig[CONFIG_WH]);
		uchar2 color = (*((device uchar2 *)&src[srcUVnum]));
		
		int nowUNum = mad24(dstConfig[CONFIG_WIDTH]/4, dstCoordinate.y, dstCoordinate.x/2);
		//nowUNum = nowUNum + dstConfig[CONFIG_WH]; 
		dst_u[nowUNum] = color.x;
		dst_v[nowUNum/*+(dstConfig[CONFIG_WH]/4)*/] = color.y;
	}
}


//NV12转I420 fast
kernel void NV12_I420(	
    device const uchar* src,	
	constant int* srcConfig,	
    device uchar* dsty,
    device uchar* dstu,
    device uchar* dstv,	
	constant int* dstConfig,
	constant float* RectConfig,
	uint2 index [[thread_position_in_grid]])	
{	
	int2 nowCoordinate;				 
    nowCoordinate.x = index.x;
    nowCoordinate.y = index.y;
	
	float2 srcCoordinate;

	srcCoordinate.x = srcConfig[CONFIG_X] + nowCoordinate.x * RectConfig[RECT_KX];
	srcCoordinate.y = srcConfig[CONFIG_Y] + nowCoordinate.y * RectConfig[RECT_KY];
		if(nowCoordinate.y >= dstConfig[CONFIG_WIDTH])
		{
		//	printf("---   nowCoordinate.y = %d, nowCoordinate.x = %d\n",nowCoordinate.y,nowCoordinate.x);
			return;
		}
	int2 dstCoordinate;	
	dstCoordinate.x = nowCoordinate.x + dstConfig[CONFIG_X];
	dstCoordinate.y = nowCoordinate.y + dstConfig[CONFIG_Y];	
	
	
	float2 decimal;
	int2 srcIntCoordinate = convert_int2(srcCoordinate);
	//decimal.x = srcCoordinate.x - srcIntCoordinate.x;
	//decimal.y = srcCoordinate.y - srcIntCoordinate.y;	
	
	
	int srcNum = mad24(srcIntCoordinate.y, srcConfig[CONFIG_WIDTH], srcIntCoordinate.x);
	int dstNum = mad24(dstCoordinate.y, dstConfig[CONFIG_WIDTH], dstCoordinate.x); 

	//dst[dstNum] = src[srcNum]; 
	
	uchar2 x00_10;
	uchar2 x01_11;
	x00_10 = (*((device uchar2 *)&src[srcNum]));
	//x01_11 = (*((device uchar2 *)&src[srcNum+srcConfig[CONFIG_WIDTH]]));
	//x00_10.x = x00_10.x + (x01_11.x - x00_10.x) * decimal.y;
	//x00_10.y = x00_10.y + (x01_11.y - x00_10.y) * decimal.y;
	//x00_10.x = x00_10.x + (x00_10.y - x00_10.x) * decimal.x;
	dsty[dstNum] = x00_10.x;
	
	if (((dstCoordinate.x % 2) == 0) && ((dstCoordinate.y % 2) == 0))
	{
	    srcIntCoordinate.x = srcIntCoordinate.x & 0xfffffffe;
		srcIntCoordinate.y = srcIntCoordinate.y & 0xfffffffe;
        
        
		int srcUVnum = mad24(srcConfig[CONFIG_HALF_W], srcIntCoordinate.y, srcIntCoordinate.x + srcConfig[CONFIG_WH]);
		uchar2 color = (*((device uchar2 *)&src[srcUVnum]));
		
		int nowUNum = mad24(dstConfig[CONFIG_WIDTH]/4, dstCoordinate.y, dstCoordinate.x/2);
		nowUNum = nowUNum;// + dstConfig[CONFIG_WH]; 
		dstu[nowUNum] = color.x;
		dstv[nowUNum/*+(dstConfig[CONFIG_WH]/4)*/] = color.y;
	}

}

//I420转NV12  fast
kernel void I420_NV12(	
    device const uchar* srcy,
	device const uchar* srcu,
	device const uchar* srcv,
	constant int* srcConfig,	
    device uchar* dst,	
	constant int* dstConfig,
	constant float* RectConfig,
	uint2 index [[thread_position_in_grid]])	
{	
	uchar2 color;

	int2 nowCoordinate;				 
    nowCoordinate.x = index.x;
    nowCoordinate.y = index.y;
	
	float2 srcCoordinate;

	if(nowCoordinate.y >= srcConfig[CONFIG_HEIGHT])
	{
		return;
	}

	if(nowCoordinate.x >= srcConfig[CONFIG_WIDTH])
	{
		return;
	}

	srcCoordinate.x = srcConfig[CONFIG_X] + nowCoordinate.x * RectConfig[RECT_KX];
	srcCoordinate.y = srcConfig[CONFIG_Y] + nowCoordinate.y * RectConfig[RECT_KY];
	
	int2 dstCoordinate;	
	dstCoordinate.x = nowCoordinate.x + dstConfig[CONFIG_X];
	dstCoordinate.y = nowCoordinate.y + dstConfig[CONFIG_Y];	
	
	
	float2 decimal;
	int2 srcIntCoordinate = convert_int2(srcCoordinate);
	//decimal.x = srcCoordinate.x - srcIntCoordinate.x;
	//decimal.y = srcCoordinate.y - srcIntCoordinate.y;	
	
	
	int srcNum = mad24(srcIntCoordinate.y, srcConfig[CONFIG_WIDTH], srcIntCoordinate.x);
	int dstNum = mad24(dstCoordinate.y, dstConfig[CONFIG_WIDTH], dstCoordinate.x); 
	//dst[dstNum] = src[srcNum]; 
	
	uchar2 x00_10;
	uchar2 x01_11;
	x00_10 = (*((device uchar2 *)&srcy[srcNum]));
	//x01_11 = (*((device uchar2 *)&src[srcNum+srcConfig[CONFIG_WIDTH]]));
	//x00_10.x = x00_10.x + (x01_11.x - x00_10.x) * decimal.y;
	//x00_10.y = x00_10.y + (x01_11.y - x00_10.y) * decimal.y;
	//x00_10.x = x00_10.x + (x00_10.y - x00_10.x) * decimal.x;
	dst[dstNum] = x00_10.x;
	if (((dstCoordinate.x % 2) == 0) && ((dstCoordinate.y % 2) == 0))
	{
	    srcIntCoordinate.x = srcIntCoordinate.x & 0xfffffffe;
		srcIntCoordinate.y = srcIntCoordinate.y & 0xfffffffe;
		int nowUNum = mad24(srcConfig[CONFIG_WIDTH]/4, srcIntCoordinate.y, srcIntCoordinate.x/2);
		nowUNum = nowUNum;// + srcConfig[CONFIG_WH]; 
		color.x = srcu[nowUNum];
		color.y = srcv[nowUNum/*+(srcConfig[CONFIG_WH]/4)*/];
		
		int dstUVnum = mad24(dstConfig[CONFIG_HALF_W], dstCoordinate.y, dstCoordinate.x + dstConfig[CONFIG_WH]);
		 *((device uchar2 *)&dst[dstUVnum]) = color;
	}
}

kernel void NV12_Remove_Padding(	
    device const uchar* src,
	int    width,
	int    height,
    device uchar* dst,	
	int    owidth,
	int    oheight,
	uint2 index [[thread_position_in_grid]])
{
    int id = index.x;
	int index_src = 0;
	int index_dst = 0;
	uchar4* dst_4 = (uchar4*)dst;
	if(id > oheight - 1)
	{
		return;
	}
	
	index_dst = id * owidth;
	
	if(id > oheight*2/3 - 1)
	{
		id += (height - oheight)*2/3;
		index_src = id * width;
	}
	else
	{
		index_src = id * width;
	}

	for(int i = (index_dst>>2); i < (index_dst + owidth)>>2 ; i++)
	{
		*(dst_4 + i) = (uchar4)vload4((index_src>>2) + i - (index_dst>>2),src);
	}
}

kernel void NV12_Add_Padding(	
    device const uchar* src,
	int    width,
	int    height,
    device uchar* dst,	
	int    owidth,
	int    oheight,
	uint2 index [[thread_position_in_grid]])
{
    int id = index.x;
	int index_src = 0;
	int index_dst = 0;
	uchar4* dst_4 = (uchar4*)dst;
	int Y_height = height*2/3;
	int Y_padding_lines = (oheight - height)*2/3;

	if(id > oheight - 1)
	{
		return;
	}

	index_dst = id * owidth;
	
	if((id >= Y_height && id < Y_height + Y_padding_lines) || id > height + Y_padding_lines - 1)
	{
		for(int i = (index_dst>>2); i < (index_dst + owidth)>>2 ; i++)
		{
			*(dst_4 + i) = (uchar4)0;
		}
		return;
	}
	else
	{
		if(id >= Y_height + Y_padding_lines)
		{
			id -= Y_padding_lines;
		}
		
		index_src = id * width;
	}

	for(int i = (index_dst>>2); i < (index_dst + width)>>2; i++)
	{
		*(dst_4 + i) = (uchar4)vload4((index_src>>2) + i - (index_dst>>2),src);
	}
	
	if(owidth > width)
	{
		for(int i = (index_dst + width)>>2; i < (index_dst + owidth)>>2 ; i++)
		{
			*(dst_4 + i) = (uchar4)0;
		}
	}
}