// Sessions as written by SOLiD (LT-mapper format) and the files Commerge reads and writes.
//   <session>/Scans/000000.pcd   keyframe cloud in the sensor frame (x y z intensity)
//   <session>/SOLiDs/000000.txt  SOLiD descriptor (line 1: R-SOLiD, line 2: A-SOLiD)
//   <session>/optimized_poses.txt, odom_poses.txt  keyframe poses (KITTI, 3x4 per row)
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include "commerge/commerge.hpp"

namespace fs = std::filesystem;

namespace commerge {
namespace {

std::string slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

std::vector<double> numbers(std::string text) {
  std::replace(text.begin(), text.end(), ',', ' ');
  std::istringstream in(text);
  std::vector<double> v;
  double x;
  while (in >> x) v.push_back(x);
  return v;
}

std::vector<std::string> sorted_files(const fs::path& dir, const std::string& ext) {
  std::vector<std::string> out;
  if (!fs::is_directory(dir)) return out;
  for (auto& e : fs::directory_iterator(dir))
    if (e.is_regular_file() && e.path().extension() == ext) out.push_back(e.path().string());
  std::sort(out.begin(), out.end());
  return out;
}

std::string lzf_decompress(const std::string& src, size_t out_len) {
  std::string out;
  out.reserve(out_len);
  size_t i = 0;
  while (i < src.size()) {
    unsigned ctrl = static_cast<unsigned char>(src[i++]);
    if (ctrl < 32) {  // literal run
      out.append(src, i, ctrl + 1);
      i += ctrl + 1;
    } else {  // back reference
      size_t len = ctrl >> 5;
      long ref = static_cast<long>(out.size()) - ((ctrl & 0x1F) << 8) - 1;
      if (len == 7) len += static_cast<unsigned char>(src[i++]);
      ref -= static_cast<unsigned char>(src[i++]);
      if (ref < 0) throw std::runtime_error("corrupt binary_compressed PCD");
      for (size_t k = 0; k < len + 2; ++k) out.push_back(out[ref++]);
    }
  }
  if (out.size() != out_len) throw std::runtime_error("corrupt binary_compressed PCD");
  return out;
}

double read_value(const char* p, char type, int size) {
  switch (type) {
    case 'F':
      if (size == 4) { float v; std::memcpy(&v, p, 4); return v; }
      if (size == 8) { double v; std::memcpy(&v, p, 8); return v; }
      break;
    case 'U':
      if (size == 1) return static_cast<unsigned char>(*p);
      if (size == 2) { uint16_t v; std::memcpy(&v, p, 2); return v; }
      if (size == 4) { uint32_t v; std::memcpy(&v, p, 4); return v; }
      break;
    case 'I':
      if (size == 1) return static_cast<signed char>(*p);
      if (size == 2) { int16_t v; std::memcpy(&v, p, 2); return v; }
      if (size == 4) { int32_t v; std::memcpy(&v, p, 4); return v; }
      break;
  }
  throw std::runtime_error("unsupported PCD field type");
}

}  // namespace

Poses read_kitti_poses(const std::string& path) {
  std::vector<double> v = numbers(slurp(path));
  if (v.size() % 12) throw std::runtime_error(path + ": not 12 values per pose");
  Poses out(v.size() / 12, Pose::Identity());
  for (size_t k = 0; k < out.size(); ++k)
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 4; ++c) out[k](r, c) = v[12 * k + 4 * r + c];
  return out;
}

void write_kitti_poses(const std::string& path, const Poses& poses) {
  std::ofstream f(path);
  char buf[32];
  for (auto& T : poses) {
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 4; ++c) {
        std::snprintf(buf, sizeof(buf), "%.9f", T(r, c));
        f << buf << ((r == 2 && c == 3) ? "\n" : " ");
      }
  }
}

Cloud read_pcd(const std::string& path) {
  std::string data = slurp(path);
  std::map<std::string, std::vector<std::string>> header;
  size_t pos = 0;
  while (pos < data.size()) {
    size_t end = data.find('\n', pos);
    if (end == std::string::npos) end = data.size();
    std::string line = data.substr(pos, end - pos);
    pos = end + 1;
    std::istringstream in(line);
    std::string key, val;
    if (!(in >> key) || key[0] == '#') continue;
    std::transform(key.begin(), key.end(), key.begin(), ::toupper);
    auto& vals = header[key];
    while (in >> val) vals.push_back(val);
    if (key == "DATA") break;
  }
  const auto& fields = header["FIELDS"];
  const size_t nf = fields.size();
  std::vector<int> size(nf), count(nf, 1), offset(nf);
  std::vector<char> type(nf);
  for (size_t k = 0; k < nf; ++k) {
    size[k] = std::stoi(header["SIZE"].at(k));
    type[k] = header["TYPE"].at(k)[0];
    if (header.count("COUNT")) count[k] = std::stoi(header["COUNT"].at(k));
  }
  int stride = 0;
  for (size_t k = 0; k < nf; ++k) offset[k] = stride, stride += size[k] * count[k];
  const long n = std::stol(header["POINTS"].at(0));
  std::string mode = header["DATA"].at(0);
  std::transform(mode.begin(), mode.end(), mode.begin(), ::tolower);
  const char* names[4] = {"x", "y", "z", "intensity"};
  int col[4] = {-1, -1, -1, -1};
  for (int c = 0; c < 4; ++c)
    for (size_t k = 0; k < nf; ++k)
      if (fields[k] == names[c]) col[c] = static_cast<int>(k);
  Cloud out = Cloud::Zero(n, 4);
  if (mode == "ascii") {
    std::istringstream in(data.substr(pos));
    std::string line;
    for (long i = 0; i < n && std::getline(in, line);) {
      std::vector<double> v = numbers(line);
      if (v.empty()) continue;
      // ascii lists every element of every field in order
      std::vector<int> first(nf);
      int idx = 0;
      for (size_t k = 0; k < nf; ++k) first[k] = idx, idx += count[k];
      for (int c = 0; c < 4; ++c)
        if (col[c] >= 0 && first[col[c]] < static_cast<int>(v.size())) out(i, c) = static_cast<float>(v[first[col[c]]]);
      ++i;
    }
  } else if (mode == "binary") {
    if (data.size() < pos + static_cast<size_t>(n) * stride) throw std::runtime_error(path + ": truncated");
    const char* base = data.data() + pos;
    for (long i = 0; i < n; ++i)
      for (int c = 0; c < 4; ++c)
        if (col[c] >= 0) out(i, c) = static_cast<float>(read_value(base + i * stride + offset[col[c]], type[col[c]], size[col[c]]));
  } else if (mode == "binary_compressed") {
    uint32_t comp, uncomp;
    std::memcpy(&comp, data.data() + pos, 4);
    std::memcpy(&uncomp, data.data() + pos + 4, 4);
    std::string raw = lzf_decompress(data.substr(pos + 8, comp), uncomp);
    // each field stored contiguously: field k at n * (sum of earlier field strides)
    for (int c = 0; c < 4; ++c) {
      if (col[c] < 0) continue;
      size_t start = static_cast<size_t>(n) * offset[col[c]];
      int elem = size[col[c]] * count[col[c]];
      for (long i = 0; i < n; ++i)
        out(i, c) = static_cast<float>(read_value(raw.data() + start + i * elem, type[col[c]], size[col[c]]));
    }
  } else {
    throw std::runtime_error(path + ": unsupported PCD DATA " + mode);
  }
  return out;
}

void write_pcd(const std::string& path, const Cloud& cloud) {
  fs::path p(path);
  if (p.has_parent_path()) fs::create_directories(p.parent_path());
  std::ofstream f(path, std::ios::binary);
  f << "# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z intensity\n"
       "SIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\nWIDTH "
    << cloud.rows() << "\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS " << cloud.rows() << "\nDATA binary\n";
  f.write(reinterpret_cast<const char*>(cloud.data()), cloud.size() * sizeof(float));
}

Matrix read_npy(const std::string& path) {
  std::string d = slurp(path);
  if (d.compare(0, 6, "\x93NUMPY") != 0) throw std::runtime_error(path + ": not a .npy file");
  size_t hlen, hstart;
  if (d[6] == 1) {
    hlen = static_cast<unsigned char>(d[8]) | (static_cast<unsigned char>(d[9]) << 8);
    hstart = 10;
  } else {
    uint32_t h;
    std::memcpy(&h, d.data() + 8, 4);
    hlen = h, hstart = 12;
  }
  std::string h = d.substr(hstart, hlen);
  auto field = [&](const std::string& key) {
    size_t p = h.find("'" + key + "'");
    if (p == std::string::npos) throw std::runtime_error(path + ": no " + key);
    return h.substr(h.find(':', p) + 1);
  };
  std::string descr = field("descr");
  descr = descr.substr(descr.find('\'') + 1);
  descr = descr.substr(0, descr.find('\''));
  if (field("fortran_order").find("True") < field("fortran_order").find(','))
    throw std::runtime_error(path + ": Fortran order not supported");
  std::string shape = field("shape");
  shape = shape.substr(shape.find('(') + 1, shape.find(')') - shape.find('(') - 1);
  std::vector<double> dims = numbers(shape);
  if (dims.size() != 2) throw std::runtime_error(path + ": expected a 2-D array");
  const long r = static_cast<long>(dims[0]), c = static_cast<long>(dims[1]);
  const char* base = d.data() + hstart + hlen;
  Matrix m(r, c);
  if (descr == "<f8") {
    std::memcpy(m.data(), base, sizeof(double) * r * c);
  } else if (descr == "<f4") {
    for (long k = 0; k < r * c; ++k) {
      float v;
      std::memcpy(&v, base + 4 * k, 4);
      m.data()[k] = v;
    }
  } else {
    throw std::runtime_error(path + ": dtype " + descr + " not supported (float32/float64)");
  }
  return m;
}

void write_npy(const std::string& path, const Matrix& m) {
  std::string h = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + std::to_string(m.rows()) + ", " +
                  std::to_string(m.cols()) + "), }";
  while ((10 + h.size() + 1) % 64) h += ' ';
  h += '\n';
  std::ofstream f(path, std::ios::binary);
  f.write("\x93NUMPY\x01\x00", 8);
  uint16_t len = static_cast<uint16_t>(h.size());
  f.write(reinterpret_cast<const char*>(&len), 2);
  f << h;
  std::vector<float> v(m.data(), m.data() + m.size());
  f.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(float));
}

Session Session::load(const std::string& root, const std::string& poses) {
  fs::path r(root);
  fs::path p;
  if (poses == "auto")
    p = fs::exists(r / "optimized_poses.txt") ? r / "optimized_poses.txt" : r / "odom_poses.txt";
  else if (poses == "optimized" || poses == "odom")
    p = r / (poses + "_poses.txt");
  else
    p = poses;
  Session s;
  s.root = root;
  s.name = fs::path(root).filename().string();
  if (s.name.empty()) s.name = fs::path(root).parent_path().filename().string();
  s.poses = read_kitti_poses(p.string());
  s.scans = sorted_files(r / "Scans", ".pcd");
  std::vector<std::string> sol = sorted_files(r / "SOLiDs", ".txt");
  std::vector<std::vector<double>> rs, as;
  for (auto& f : sol) {
    std::istringstream in(slurp(f));
    std::string l1, l2;
    std::getline(in, l1);
    rs.push_back(numbers(l1));
    if (std::getline(in, l2)) {
      std::vector<double> v = numbers(l2);
      if (!v.empty()) as.push_back(v);
    }
  }
  auto to_matrix = [](const std::vector<std::vector<double>>& rows) {
    if (rows.empty()) return Matrix();
    Matrix m(rows.size(), rows[0].size());
    for (size_t i = 0; i < rows.size(); ++i) {
      if (rows[i].size() != rows[0].size()) throw std::runtime_error("SOLiDs/: descriptors of different lengths");
      for (size_t j = 0; j < rows[i].size(); ++j) m(i, j) = rows[i][j];
    }
    return m;
  };
  s.rsolid = to_matrix(rs);
  s.asolid = to_matrix(as);
  if (!s.scans.empty() && s.scans.size() != s.poses.size())
    throw std::runtime_error(root + ": " + std::to_string(s.scans.size()) + " scans but " +
                             std::to_string(s.poses.size()) + " poses");
  return s;
}

Positions Session::positions() const {
  Positions p(poses.size(), 3);
  for (size_t k = 0; k < poses.size(); ++k) p.row(k) = poses[k].block<3, 1>(0, 3).transpose();
  return p;
}

Cloud Session::cloud_in_world(int k, double min_range) const {
  Cloud pts = read_pcd(scans.at(k));
  const Pose& T = poses.at(k);
  Cloud out(pts.rows(), 4);
  long n = 0;
  for (long i = 0; i < pts.rows(); ++i) {
    Eigen::Vector3d p = pts.row(i).head<3>().cast<double>().transpose();
    if (min_range > 0 && !(p.norm() > min_range)) continue;
    Eigen::Vector3d q = T.block<3, 3>(0, 0) * p + T.block<3, 1>(0, 3);
    out.row(n) << static_cast<float>(q.x()), static_cast<float>(q.y()), static_cast<float>(q.z()), pts(i, 3);
    ++n;
  }
  out.conservativeResize(n, 4);
  return out;
}

Cloud voxel_downsample(const Cloud& cloud, double leaf) {
  if (leaf <= 0 || cloud.rows() == 0) return cloud;
  struct Hash {
    size_t operator()(const std::array<int64_t, 3>& k) const {
      return std::hash<int64_t>()(k[0] * 73856093 ^ k[1] * 19349663 ^ k[2] * 83492791);
    }
  };
  std::unordered_set<std::array<int64_t, 3>, Hash> seen;
  Cloud out(cloud.rows(), 4);
  long n = 0;
  for (long i = 0; i < cloud.rows(); ++i) {
    std::array<int64_t, 3> key;
    for (int c = 0; c < 3; ++c) key[c] = static_cast<int64_t>(std::floor(cloud(i, c) / leaf));
    if (seen.insert(key).second) out.row(n++) = cloud.row(i);
  }
  out.conservativeResize(n, 4);
  return out;
}

Cloud build_submap(const Session& s, const std::vector<int>& keyframes, double min_range, double voxel) {
  std::vector<Cloud> clouds;
  long total = 0;
  for (int k : keyframes) {
    clouds.push_back(s.cloud_in_world(k, min_range));
    total += clouds.back().rows();
  }
  Cloud out(total, 4);
  long row = 0;
  for (auto& c : clouds) {
    out.middleRows(row, c.rows()) = c;
    row += c.rows();
  }
  return voxel_downsample(out, voxel);
}

}  // namespace commerge
