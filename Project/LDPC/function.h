#pragma once

#include "../LDPC_Lib/LdpcLib.h"

#pragma comment(lib, "../x64/Release/LDPC_Lib.lib")
//#pragma comment(lib, "../x64/Debug/LDPC_Lib.lib")

void DisplayHelp(void);

void Main_CPU(int argc, char* argv[]);

void Main_GPU(int argc, char* argv[]);