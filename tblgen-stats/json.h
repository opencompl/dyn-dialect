//===- json.h - JSON helpers ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines helpers used in the op generators.
//
//===----------------------------------------------------------------------===//

#ifndef DYN_DIALECT_JSON_H_
#define DYN_DIALECT_JSON_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {

struct JSON {
  virtual void print(raw_ostream &os) = 0;

  virtual ~JSON(){};
};

struct JSONDict : JSON {
  llvm::StringMap<std::unique_ptr<JSON>> value;

  ~JSONDict() override{};

  void print(raw_ostream &os) override {
    os << "{";
    llvm::interleaveComma(value, os, [&os](auto &item) {
      os << '"' << item.first() << '"' << ": ";
      item.second->print(os);
    });
    os << "}";
  }

  static std::unique_ptr<JSONDict> get() {
    return std::make_unique<JSONDict>();
  }

  void insert(StringRef key, std::unique_ptr<JSON> &&val) {
    value.insert({key, std::move(val)});
  }
};

struct JSONList : JSON {
  std::vector<std::unique_ptr<JSON>> value;

  ~JSONList() override{};

  void print(raw_ostream &os) override {
    os << "[";
    llvm::interleaveComma(value, os, [&os](auto &item) {
      item->print(os);
    });
    os << "]";
  }

  void insert(std::unique_ptr<JSON>&& item) {
    value.push_back(std::move(item));
  }

  static std::unique_ptr<JSONList> get() {
    return std::make_unique<JSONList>();
  }
};

struct JSONBool : JSON {
  bool value;

  JSONBool(bool value) : value(value) {}
  ~JSONBool() override{};

  void print(raw_ostream &os) override {
    if (value) {
      os << "true";
    } else {
      os << "false";
    }
  }

  static std::unique_ptr<JSONBool> get(bool value) {
    return std::make_unique<JSONBool>(value);
  }
};

struct JSONInt : JSON {
  int value;

  JSONInt(int value) : value(value) {}
  ~JSONInt() override{};

  void print(raw_ostream &os) override { os << value; }

  static std::unique_ptr<JSONInt> get(int value) {
    return std::make_unique<JSONInt>(value);
  }
};

struct JSONStr : JSON {
  std::string value;

  JSONStr(StringRef value) : value(value.str()) {}
  ~JSONStr() override{};

  void print(raw_ostream &os) override { os << '"' << value << '"'; }

  static std::unique_ptr<JSONStr> get(StringRef value) {
    return std::make_unique<JSONStr>(value);
  }
};

} // namespace mlir

#endif // DYN_DIALECT_JSON_H_
