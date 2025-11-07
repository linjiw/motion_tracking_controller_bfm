#include "motion_tracking_controller/bfm_support/NpzReader.h"

#include <zlib.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace legged::bfm {
namespace {

constexpr uint32_t kCentralDirMagic = 0x02014b50;
constexpr uint32_t kLocalFileMagic = 0x04034b50;
constexpr uint32_t kEndOfCentralDirMagic = 0x06054b50;

uint16_t read_u16(const uint8_t* ptr) {
  return static_cast<uint16_t>(ptr[0]) | (static_cast<uint16_t>(ptr[1]) << 8);
}

uint32_t read_u32(const uint8_t* ptr) {
  return static_cast<uint32_t>(ptr[0]) | (static_cast<uint32_t>(ptr[1]) << 8) |
         (static_cast<uint32_t>(ptr[2]) << 16) | (static_cast<uint32_t>(ptr[3]) << 24);
}

struct ZipEntry {
  std::string name;
  uint32_t compressed_size{0};
  uint32_t uncompressed_size{0};
  uint32_t local_header_offset{0};
  uint16_t compression_method{0};
  uint16_t flags{0};
};

bool parse_end_of_central_directory(const std::vector<uint8_t>& buffer,
                                    uint32_t& directory_offset,
                                    uint16_t& entry_count) {
  const size_t size = buffer.size();
  if (size < 22) {
    throw std::runtime_error("NPZ file too small to contain central directory.");
  }

  size_t eocd_offset = std::string::npos;
  for (size_t i = 0; i + 21 < size; ++i) {
    if (read_u32(buffer.data() + i) == kEndOfCentralDirMagic) {
      eocd_offset = i;
    }
  }
  if (eocd_offset == std::string::npos) {
    throw std::runtime_error("Failed to locate ZIP end-of-central-directory record.");
  }

  const uint8_t* rec = buffer.data() + eocd_offset;
  entry_count = read_u16(rec + 10);
  directory_offset = read_u32(rec + 16);
  return true;
}

std::vector<ZipEntry> parse_central_directory(const std::vector<uint8_t>& buffer,
                                              uint32_t directory_offset,
                                              uint16_t entry_count) {
  size_t offset = directory_offset;
  std::vector<ZipEntry> entries;
  entries.reserve(entry_count);
  const size_t size = buffer.size();
  for (uint16_t idx = 0; idx < entry_count; ++idx) {
    if (offset + 46 > size || read_u32(buffer.data() + offset) != kCentralDirMagic) {
      throw std::runtime_error("Malformed ZIP central directory.");
    }
    const uint8_t* header = buffer.data() + offset;
    ZipEntry entry;
    entry.flags = read_u16(header + 8);
    entry.compression_method = read_u16(header + 10);
    entry.compressed_size = read_u32(header + 20);
    entry.uncompressed_size = read_u32(header + 24);
    const uint16_t name_len = read_u16(header + 28);
    const uint16_t extra_len = read_u16(header + 30);
    const uint16_t comment_len = read_u16(header + 32);
    entry.local_header_offset = read_u32(header + 42);

    if (offset + 46 + name_len + extra_len + comment_len > size) {
      throw std::runtime_error("Central directory entry exceeds file size.");
    }
    entry.name.assign(reinterpret_cast<const char*>(header + 46), name_len);
    entries.push_back(entry);
    offset += 46 + name_len + extra_len + comment_len;
  }
  return entries;
}

ZipEntry find_entry(const std::vector<ZipEntry>& entries, const std::string& target) {
  for (const auto& entry : entries) {
    if (entry.name == target) {
      return entry;
    }
  }
  throw std::runtime_error("Array \"" + target + "\" not found in NPZ archive.");
}

std::vector<uint8_t> extract_entry(const std::vector<uint8_t>& buffer, const ZipEntry& entry) {
  if (entry.flags & 0x0008) {
    throw std::runtime_error("ZIP entries with streaming data descriptors are not supported.");
  }
  const size_t size = buffer.size();
  if (entry.local_header_offset + 30 > size ||
      read_u32(buffer.data() + entry.local_header_offset) != kLocalFileMagic) {
    throw std::runtime_error("Malformed ZIP local file header.");
  }
  const uint8_t* local = buffer.data() + entry.local_header_offset;
  const uint16_t name_len = read_u16(local + 26);
  const uint16_t extra_len = read_u16(local + 28);
  const size_t data_offset = entry.local_header_offset + 30 + name_len + extra_len;
  if (data_offset + entry.compressed_size > size) {
    throw std::runtime_error("Compressed data outside file bounds.");
  }

  const uint8_t* compressed = buffer.data() + data_offset;
  std::vector<uint8_t> output(entry.uncompressed_size);

  if (entry.compression_method == 0) {
    if (entry.uncompressed_size != entry.compressed_size) {
      throw std::runtime_error("Stored ZIP entry size mismatch.");
    }
    std::memcpy(output.data(), compressed, entry.compressed_size);
    return output;
  }

  if (entry.compression_method != 8) {
    throw std::runtime_error("Unsupported ZIP compression method.");
  }

  uLongf dest_len = entry.uncompressed_size;
  const int status = uncompress(output.data(), &dest_len, compressed, entry.compressed_size);
  if (status != Z_OK || dest_len != entry.uncompressed_size) {
    throw std::runtime_error("Failed to decompress ZIP entry.");
  }
  return output;
}

struct NpyHeader {
  size_t element_count{0};
  size_t dtype_size{0};
};

NpyHeader parse_npy_header(const std::vector<uint8_t>& buffer) {
  if (buffer.size() < 10) {
    throw std::runtime_error("NPY file too small.");
  }
  if (buffer[0] != 0x93 || buffer[1] != 'N' || buffer[2] != 'U' || buffer[3] != 'M' || buffer[4] != 'P' ||
      buffer[5] != 'Y') {
    throw std::runtime_error("Invalid NPY magic.");
  }
  const uint16_t header_len = read_u16(buffer.data() + 8);
  const size_t header_start = 10;
  const size_t header_end = header_start + header_len;
  if (header_end > buffer.size()) {
    throw std::runtime_error("NPY header truncated.");
  }
  const std::string header(reinterpret_cast<const char*>(buffer.data() + header_start), header_len);

  const auto find = [&](const std::string& key) -> size_t { return header.find(key); };
  const size_t descr_pos = find("'descr'");
  const size_t shape_pos = find("'shape'");
  if (descr_pos == std::string::npos || shape_pos == std::string::npos) {
    throw std::runtime_error("NPY header missing descr/shape.");
  }

  const size_t descr_start = header.find('\'', descr_pos + 6);
  const size_t descr_end = header.find('\'', descr_start + 1);
  if (descr_start == std::string::npos || descr_end == std::string::npos) {
    throw std::runtime_error("Failed to parse NPY dtype.");
  }
  const std::string dtype = header.substr(descr_start + 1, descr_end - descr_start - 1);
  size_t dtype_size = 0;
  if (dtype == "<f4" || dtype == "|f4" || dtype == "=f4") {
    dtype_size = 4;
  } else if (dtype == "<f8" || dtype == "|f8" || dtype == "=f8") {
    dtype_size = 8;
  } else {
    throw std::runtime_error("Unsupported NPY dtype: " + dtype);
  }

  const size_t shape_start = header.find('(', shape_pos);
  const size_t shape_end = header.find(')', shape_start);
  if (shape_start == std::string::npos || shape_end == std::string::npos || shape_end <= shape_start + 1) {
    throw std::runtime_error("Failed to parse NPY shape.");
  }

  const std::string shape_body = header.substr(shape_start + 1, shape_end - shape_start - 1);
  size_t count = 1;
  size_t begin = 0;
  while (begin < shape_body.size()) {
    size_t end = shape_body.find(',', begin);
    if (end == std::string::npos) {
      end = shape_body.size();
    }
    const std::string token = shape_body.substr(begin, end - begin);
    const std::string trimmed = token.substr(token.find_first_not_of(" \t"), token.find_last_not_of(" \t") -
                                                                                  token.find_first_not_of(" \t") + 1);
    if (!trimmed.empty()) {
      count *= static_cast<size_t>(std::stoll(trimmed));
    }
    begin = end + 1;
  }

  return NpyHeader{count, dtype_size};
}

}  // namespace

NpzArray readArray(const std::string& npz_path, const std::string& array_name) {
  std::ifstream input(npz_path, std::ios::binary);
  if (!input.is_open()) {
    throw std::runtime_error("Failed to open NPZ file: " + npz_path);
  }
  std::vector<uint8_t> buffer((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());

  uint32_t dir_offset = 0;
  uint16_t entry_count = 0;
  parse_end_of_central_directory(buffer, dir_offset, entry_count);
  const auto entries = parse_central_directory(buffer, dir_offset, entry_count);
  const auto entry = find_entry(entries, array_name);
  const auto raw = extract_entry(buffer, entry);
  const auto header = parse_npy_header(raw);

  const size_t data_offset = 10 + read_u16(raw.data() + 8);
  if (data_offset + header.dtype_size * header.element_count > raw.size()) {
    throw std::runtime_error("NPY payload truncated.");
  }

  NpzArray out;
  out.count = header.element_count;
  out.data.resize(header.element_count);
  const uint8_t* src = raw.data() + data_offset;
  if (header.dtype_size == 4) {
    for (size_t i = 0; i < header.element_count; ++i) {
      float value;
      std::memcpy(&value, src + i * 4, 4);
      out.data[i] = value;
    }
  } else if (header.dtype_size == 8) {
    for (size_t i = 0; i < header.element_count; ++i) {
      double value;
      std::memcpy(&value, src + i * 8, 8);
      out.data[i] = static_cast<float>(value);
    }
  } else {
    throw std::runtime_error("Unsupported dtype size in NPZ array.");
  }
  return out;
}

}  // namespace legged::bfm
