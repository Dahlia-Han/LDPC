#include <stdio.h>
#include "function.h"
#include <string.h>



int main(int argc, char* argv[])
{
	// �ж�������������Ƿ�����
	if (argc == 1)
	{
		printf("��֧���޲�������,���������,ͨ��\"-help\"������ȡ����\n");
		DisplayHelp();
		printf("�밴<Enter>���˳�...\n");
		char c = getchar();
		return 0;
	}


	// ��ӡ����
	if (strcmp(argv[1], "-help") == 0)
	{
		DisplayHelp();
		printf("�밴<Enter>���˳�...\n");
		char c = getchar();
		return 0;
	}

	// ��ȡ�������
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