#pragma once

#include <torch/extension.h>

#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <malloc.h>

#include <iostream>
#include <cstdlib>
#include <stdexcept>
#include <string>

// DIRECT IO routines
py::bytes load_file_direct(const std::string& filename, const size_t& blocksize, size_t filesize=0);

size_t save_file_direct(const std::string& filename, const std::string& data,
			const size_t& blocksize, const bool& sync=true);

