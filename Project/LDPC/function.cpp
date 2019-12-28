#include "function.h"
#include <stdio.h>
#include <fstream>
#include <time.h>

void DisplayHelp(void)
{
	printf("输入参数说明:\n");
	printf("\n");

	printf("常用参数:\n");
	printf("argv[1]	            - 校验矩阵H位置\n");
	printf("argv[2]             - 输出文件\n");
	printf("-Seed               -噪声生成随机种子，默认值0\n");
	printf("-ExitMethod         -仿真退出方式，0-以Frame为准，1-以Bit为准，默认值0\n");
	printf("-MinError           -最小错误数，默认值100\n");
	printf("-MaxTrans           -最大传输数，默认值1000000\n");
	printf("-Rate               -指定码率，默认值由矩阵行列值计算\n");
	printf("-PBitNum            -打掉信息位比特数，默认值0\n");
	printf("-MaxITR             -最大迭代次数，默认值20\n");
	printf("-MinEbN0indB        -最小信噪比，默认值-2dB\n");
	printf("-MaxEbN0indB        -最大信噪比，默认值20dB\n");
	printf("-InvEbN0indB        -信噪比增量，默认值0.2dB\n");
	printf("-DecodeMethod       -指定译码方法：LogBP,BP\n");

	printf("\n");

	printf("CPU仿真参数:\n");
	printf("-PreExitBMI         -提前退出, 0 - false; 1 - true\n");
	printf("\n");

	printf("GPU仿真参数:\n");
	printf("-GPU                -使用GPU加速\n");
	printf("-MemoryLimit        -使用GPU显存限制\n");
	printf("\n");
}

void Main_CPU(int argc, char* argv[])
{
	// 仿真参数初始化
	char* InputFileName = argv[1];
	char* OutputFileName = argv[2];
	uint64_t Seed = 0;
	uint32_t ExitMethod = 0;
	uint64_t MinError = 100;
	uint64_t MaxTrans = 1e6;
	float Rate = -1;
	uint64_t NumPunchBits = 0;
	uint32_t MaxITR = 20;
	uint32_t isExitBeforeMaxItr = false;
	float MinEbN0indB = -2;
	float MaxEbN0indB = 20;
	float InvEbN0indB = 0.2;
	char* DecodeMethod = nullptr;

	// 读取输入参数
	for (uint32_t i = 3; i < argc; i++)
	{
		// Seed
		if (strcmp(argv[i], "-Seed") == 0)
			Seed = _atoi64(argv[i + 1]);
		// ExitMethod
		else if (strcmp(argv[i], "-ExitMethod") == 0)
			ExitMethod = _atoi64(argv[i + 1]);
		// MinError
		else if (strcmp(argv[i], "-MinError") == 0)
			MinError = _atoi64(argv[i + 1]);
		// MaxTrans
		else if (strcmp(argv[i], "-MaxTrans") == 0)
			MaxTrans = _atoi64(argv[i + 1]);
		// Rate
		else if (strcmp(argv[i], "-Rate") == 0)
			Rate = atof(argv[i + 1]);
		// PunchBitNum
		else if (strcmp(argv[i], "-PBitNum") == 0)
			NumPunchBits = _atoi64(argv[i + 1]);
		// MaxITR
		else if (strcmp(argv[i], "-MaxITR") == 0)
			MaxITR = _atoi64(argv[i + 1]);
		// isExitBeforeMaxItr
		else if (strcmp(argv[i], "-MaxITR") == 0)
			isExitBeforeMaxItr = _atoi64(argv[i + 1]);
		// MinEbN0indB
		else if (strcmp(argv[i], "-MinEbN0indB") == 0)
			MinEbN0indB = atof(argv[i + 1]);
		// MaxEbN0indB
		else if (strcmp(argv[i], "-MaxEbN0indB") == 0)
			MaxEbN0indB = atof(argv[i + 1]);
		// InvEbN0indB
		else if (strcmp(argv[i], "-InvEbN0indB") == 0)
			InvEbN0indB = atof(argv[i + 1]);
		// DecodeMethod
		else if (strcmp(argv[i], "-DecodeMethod") == 0)
			DecodeMethod = argv[i + 1];
	}

	CheckMatrix matrix;
	if (LoadCheckMatrix(InputFileName, matrix) == false)
	{
		printf("矩阵文件不存在.\n");
		return;
	}
	if (Rate <= 0)
		Rate = (matrix.Col - NumPunchBits - matrix.Row) / (double)matrix.Col;


	uint64_t ErrorFrames = 0;
	uint64_t TransFrames = 0;
	uint64_t ErrorBits = 0;
	uint64_t TransBits = 0;
	std::ofstream outFile(argv[2], std::ios::app);
	for (float ebn0 = MinEbN0indB; ebn0 <= MaxEbN0indB; ebn0 += InvEbN0indB)
	{
		time_t Point_Time = clock();
		if (DecodeMethod == nullptr)
			break;

		// BP译码算法
		else if (strcmp(DecodeMethod, "BP") == 0)
		{
			LdpcSimulation_BP(matrix, Seed, ebn0, Rate, MinError, MaxTrans, MaxITR, NumPunchBits, isExitBeforeMaxItr, ExitMethod,
				ErrorBits, TransBits, ErrorFrames, TransFrames);
			printf("EbN0=%.2f, ErrorBits=%I64d, TransBits=%I64d, ErrorFrames=%I64d, TransFrames=%I64d, BER=%.2e, FER=%.2e, BP, Time=%.1fs\n", ebn0,
				ErrorBits, TransBits, ErrorFrames, TransFrames, ErrorBits / (double)TransBits, ErrorFrames / (double)TransFrames, (clock() - Point_Time) / (double)CLOCKS_PER_SEC);
		}
		else if (strcmp(DecodeMethod, "LogBP") == 0)
		{
			LdpcSimulation_LogBP(matrix, Seed, ebn0, Rate, MinError, MaxTrans, MaxITR, NumPunchBits, isExitBeforeMaxItr, ExitMethod,
				ErrorBits, TransBits, ErrorFrames, TransFrames);
			printf("EbN0=%.2f, ErrorBits=%I64d, TransBits=%I64d, ErrorFrames=%I64d, TransFrames=%I64d, BER=%.2e, FER=%.2e, LogBP, Time=%.1fs\n", ebn0,
				ErrorBits, TransBits, ErrorFrames, TransFrames, ErrorBits / (double)TransBits, ErrorFrames / (double)TransFrames, (clock() - Point_Time) / (double)CLOCKS_PER_SEC);
		}
		else
		{
			printf("尚未实现该算法，请重新确认...\n");
			break;
		}


		// 输出结果
		float BER = ErrorBits / (double)TransBits;
		float FER = ErrorFrames / (double)TransFrames;
		outFile << ebn0 << "\t" << ErrorBits << "\t" << TransBits << "\t"
			<< ErrorFrames << "\t" << TransFrames << "\t" << BER << "\t" << FER << "\n";
		outFile.flush();

		// 判断是否符合退出条件
		if (ExitMethod == 0)
		{
			if (ErrorFrames < MinError)
				break;
			if (TransFrames >= MaxTrans)
				break;
		}
		else
		{
			if (ErrorBits < MinError)
				break;
			if (TransBits >= MaxTrans)
				break;
		}
	}
	outFile.close();

}

void Main_GPU(int argc, char* argv[])
{
	// 仿真参数初始化
	char* InputFileName = argv[1];
	char* OutputFileName = argv[2];
	uint64_t Seed = 0;
	uint32_t ExitMethod = 0;
	uint64_t MinError = 100;
	uint64_t MaxTrans = 1e6;
	float Rate = -1;
	uint64_t NumPunchBits = 0;
	uint32_t MaxITR = 20;
	float MinEbN0indB = -2;
	float MaxEbN0indB = 20;
	float InvEbN0indB = 0.2;
	char* DecodeMethod = nullptr;
	float MemoryLimit = 50;

	// 读取输入参数
	for (uint32_t i = 3; i < argc; i++)
	{
		// Seed
		if (strcmp(argv[i], "-Seed") == 0)
			Seed = _atoi64(argv[i + 1]);
		// ExitMethod
		else if (strcmp(argv[i], "-ExitMethod") == 0)
			ExitMethod = _atoi64(argv[i + 1]);
		// MinError
		else if (strcmp(argv[i], "-MinError") == 0)
			MinError = _atoi64(argv[i + 1]);
		// MaxTrans
		else if (strcmp(argv[i], "-MaxTrans") == 0)
			MaxTrans = _atoi64(argv[i + 1]);
		// Rate
		else if (strcmp(argv[i], "-Rate") == 0)
			Rate = atof(argv[i + 1]);
		// PunchBitNum
		else if (strcmp(argv[i], "-PBitNum") == 0)
			NumPunchBits = _atoi64(argv[i + 1]);
		// MaxITR
		else if (strcmp(argv[i], "-MaxITR") == 0)
			MaxITR = _atoi64(argv[i + 1]);
		// MinEbN0indB
		else if (strcmp(argv[i], "-MinEbN0indB") == 0)
			MinEbN0indB = atof(argv[i + 1]);
		// MaxEbN0indB
		else if (strcmp(argv[i], "-MaxEbN0indB") == 0)
			MaxEbN0indB = atof(argv[i + 1]);
		// InvEbN0indB
		else if (strcmp(argv[i], "-InvEbN0indB") == 0)
			InvEbN0indB = atof(argv[i + 1]);
		// MemoryLimit
		else if (strcmp(argv[i], "-MemoryLimit") == 0)
			MemoryLimit = atof(argv[i + 1]);
		// DecodeMethod
		else if (strcmp(argv[i], "-DecodeMethod") == 0)
			DecodeMethod = argv[i + 1];
	}

	CheckMatrix_GPU matrix;
	if (LoadCheckMatrix_GPU(InputFileName, matrix) == false)
	{
		printf("矩阵文件不存在.\n");
		return;
	}
	if (Rate <= 0)
		Rate = (matrix.Col - NumPunchBits - matrix.Row) / (double)matrix.Col;


	uint64_t ErrorFrames = 0;
	uint64_t TransFrames = 0;
	uint64_t ErrorBits = 0;
	uint64_t TransBits = 0;
	std::ofstream outFile(argv[2], std::ios::app);
	for (float ebn0 = MinEbN0indB; ebn0 <= MaxEbN0indB; ebn0 += InvEbN0indB)
	{
		time_t Point_Time = clock();
		if (DecodeMethod == nullptr)
			break;

		// BP译码算法
		else if (strcmp(DecodeMethod, "BP") == 0)
		{
			LdpcSimulation_GPU_BP(matrix, Seed, ebn0, Rate, MinError, MaxTrans, MaxITR, NumPunchBits, ExitMethod, MemoryLimit,
				ErrorBits, TransBits, ErrorFrames, TransFrames);
			printf("EbN0=%.2f, ErrorBits=%I64d, TransBits=%I64d, ErrorFrames=%I64d, TransFrames=%I64d, BER=%.2e, FER=%.2e, BP, Time=%.1fs\n", ebn0,
				ErrorBits, TransBits, ErrorFrames, TransFrames, ErrorBits / (double)TransBits, ErrorFrames / (double)TransFrames, (clock() - Point_Time) / (double)CLOCKS_PER_SEC);
		}
		/*else if (strcmp(DecodeMethod, "LogBP") == 0)
		{
			LdpcSimulation_LogBP(matrix, Seed, ebn0, Rate, MinError, MaxTrans, MaxITR, NumPunchBits, isExitBeforeMaxItr, ExitMethod,
				ErrorBits, TransBits, ErrorFrames, TransFrames);
			printf("EbN0=%.2f, ErrorBits=%I64d, TransBits=%I64d, ErrorFrames=%I64d, TransFrames=%I64d, BER=%.2e, FER=%.2e, LogBP, Time=%.1fs\n", ebn0,
				ErrorBits, TransBits, ErrorFrames, TransFrames, ErrorBits / (double)TransBits, ErrorFrames / (double)TransFrames, (clock() - Point_Time) / (double)CLOCKS_PER_SEC);
		}*/
		else
		{
			printf("尚未实现该算法，请重新确认...\n");
			break;
		}


		// 输出结果
		float BER = ErrorBits / (double)TransBits;
		float FER = ErrorFrames / (double)TransFrames;
		outFile << ebn0 << "\t" << ErrorBits << "\t" << TransBits << "\t"
			<< ErrorFrames << "\t" << TransFrames << "\t" << BER << "\t" << FER << "\n";
		outFile.flush();

		// 判断是否符合退出条件
		if (ExitMethod == 0)
		{
			if (ErrorFrames < MinError)
				break;
			if (TransFrames >= MaxTrans)
				break;
		}
		else
		{
			if (ErrorBits < MinError)
				break;
			if (TransBits >= MaxTrans)
				break;
		}
	}
	outFile.close();
}
