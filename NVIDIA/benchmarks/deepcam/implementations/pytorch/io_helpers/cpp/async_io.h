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
#include <vector>
#include <string>
#include <memory>
#include <tuple>
#include <queue>

#include <libaio.h>


typedef std::shared_ptr<iocb> iocb_ptr;

class aio_handler {
 public:
  // constructor type
  aio_handler(const size_t& max_events);
  ~aio_handler();

  // submit IO
  void submit_save_file_direct(const::std::string& filename, const std::string& data, const size_t& blocksize);

  // wait for IO
  void wait_all_save_file_direct();
  
 protected:
  size_t _max_events;
  std::vector<io_event> _events;
  io_context_t _write_ctx;
  io_context_t _read_ctx;
  std::queue<iocb_ptr> _write_queue;
};
