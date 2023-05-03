// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file HFMLResponse.h
/// \brief Class to compute the ML response for HF-analysis selections
/// \author Fabio Catalano <fabio.catalano@cern.ch>, Universita' and INFN Torino

#ifndef O2_ANALYSIS_HFMLRESPONSE_H_
#define O2_ANALYSIS_HFMLRESPONSE_H_

#include <vector>
#include <string>

#include <onnxruntime/core/session/experimental_onnxruntime_cxx_api.h>

#include "Framework/Array2D.h"
#include "PWGHF/Core/HFSelectorCuts.h"

namespace o2::analysis
{
class HFMLResponse
{
  public:
    HFMLResponse() = default;
    virtual ~HFMLResponse() = default;

  private:
    std::vector<double> mBinsLimits = {}; // bin limits of the variable (e.g. pT) used to select which model to use
    std::vector<std::string> mPaths = {}; // paths to the models, one for each bin
    std::vector<o2::analysis::hf_cuts_ml::CutDirection> mCutDir = {}; // direction of the cuts on the model scores (no cut is also supported)
    o2::framework::LabeledArray<double> mCuts = {}; // array of cut values to apply on the model scores
};


}  // namespace o2::analysis

#endif // O2_ANALYSIS_HFMLRESPONSE_H_
