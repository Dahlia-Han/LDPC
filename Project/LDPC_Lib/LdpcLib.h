#pragma once

#include "CheckMatrix.h"

/************************************************************************************************************************/
// 添加高斯白噪声
void AddAWGN(const uint32_t CodeBitLength, const float R, const float sigma, const uint8_t *Code, float* Signal);

// Punch Bits
void PunchBits(const uint32_t Frames, const uint32_t CodeBitLength, const uint32_t BitsPunchedPerFrame, float* Signal);

/************************************************************************************************************************/
// BP译码方法
void LdpcDecode_BP(const CheckMatrix& matrix, const float sigma, const uint32_t MaxItr, const bool isExitBeforeMaxItr,
	const float* Signal, uint8_t* Decode);

// LogBP译码方法
void LdpcDecode_LogBP(const CheckMatrix& matrix, const float sigma, const uint32_t MaxItr, const bool isExitBeforeMaxItr,
	const float* Signal, uint8_t* Decode);


/************************************************************************************************************************/
// LDPC 校验矩阵 性能仿真
extern "C" _declspec(dllexport)
bool LdpcSimulation_BP(const CheckMatrix& matrix, const uint64_t Seed, const float EbN0indB, const float Rate,
	const uint64_t MinError, const uint64_t MaxTrans, const uint32_t MaxITR, const uint64_t NumPunchBits,
	const bool isExitBeforeMaxItr, const uint32_t ExitMethod,
	uint64_t& ErrorBits, uint64_t& TransBits, uint64_t& ErrorFrames, uint64_t& TransFrames);

extern "C" _declspec(dllexport)
bool LdpcSimulation_LogBP(const CheckMatrix & matrix, const uint64_t Seed, const float EbN0indB, const float Rate,
	const uint64_t MinError, const uint64_t MaxTrans, const uint32_t MaxITR, const uint64_t NumPunchBits,
	const bool isExitBeforeMaxItr, const uint32_t ExitMethod,
	uint64_t & ErrorBits, uint64_t & TransBits, uint64_t & ErrorFrames, uint64_t & TransFrames);




