#include <stdio.h>
#include "function.h"
#include <string.h>



int main(int argc, char* argv[])
{
	// 判断输入参数数量是否正常
	if (argc == 1)
	{
		printf("不支持无参数运行,请输入参数,通过\"-help\"参数获取帮助\n");
		DisplayHelp();
		printf("请按<Enter>键退出...\n");
		char c = getchar();
		return 0;
	}


	// 打印帮助
	if (strcmp(argv[1], "-help") == 0)
	{
		DisplayHelp();
		printf("请按<Enter>键退出...\n");
		char c = getchar();
		return 0;
	}

	// 读取输入参数
	uint32_t isGPU = 0;
	for (uint32_t i = 3; i < argc; i++)
	{
		if (strcmp(argv[i], "-GPU") == 0)
			isGPU = 1;
	}

	if (isGPU == 0)
	{
		Main_CPU(argc, argv);
	}
	else
	{
		Main_GPU(argc, argv);
	}


	return 0;
}