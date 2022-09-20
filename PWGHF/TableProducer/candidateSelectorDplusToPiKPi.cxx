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

/// \file candidateSelectorDplusToPiKPi.cxx
/// \brief D± → π± K∓ π± selection task
///
/// \author Fabio Catalano <fabio.catalano@cern.ch>, Politecnico and INFN Torino
/// \author Vít Kučera <vit.kucera@cern.ch>, CERN

#include <onnxruntime/core/session/experimental_onnxruntime_cxx_api.h>
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "PWGHF/DataModel/CandidateReconstructionTables.h"
#include "PWGHF/DataModel/CandidateSelectionTables.h"
#include "Common/Core/TrackSelectorPID.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::aod::hf_cand_3prong;
using namespace o2::analysis::hf_cuts_dplus_to_pi_k_pi;

/// Struct for applying Dplus to piKpi selection cuts
struct HfCandidateSelectorDplusToPiKPi {
  Produces<aod::HfSelDplusToPiKPi> hfSelDplusToPiKPiCandidate;
  Produces<aod::HfMlDplusToPiKPi> hfMlDplusToPiKPiCandidate;

  Configurable<double> ptCandMin{"ptCandMin", 1., "Lower bound of candidate pT"};
  Configurable<double> ptCandMax{"ptCandMax", 36., "Upper bound of candidate pT"};
  // TPC PID
  Configurable<double> ptPidTpcMin{"ptPidTpcMin", 0.15, "Lower bound of track pT for TPC PID"};
  Configurable<double> ptPidTpcMax{"ptPidTpcMax", 20., "Upper bound of track pT for TPC PID"};
  Configurable<double> nSigmaTpcMax{"nSigmaTpcMax", 3., "Nsigma cut on TPC"};
  // TOF PID
  Configurable<double> ptPidTofMin{"ptPidTofMin", 0.15, "Lower bound of track pT for TOF PID"};
  Configurable<double> ptPidTofMax{"ptPidTofMax", 20., "Upper bound of track pT for TOF PID"};
  Configurable<double> nSigmaTofMax{"nSigmaTofMax", 3., "Nsigma cut on TOF"};
  // topological cuts
  Configurable<std::vector<double>> binsPt{"binsPt", std::vector<double>{hf_cuts_dplus_to_pi_k_pi::vecBinsPt}, "pT bin limits"};
  Configurable<LabeledArray<double>> cuts{"cuts", {hf_cuts_dplus_to_pi_k_pi::cuts[0], nBinsPt, nCutVars, labelsPt, labelsCutVar}, "Dplus candidate selection per pT bin"};

  // ML inference
  Configurable<bool> b_applyML{"b_applyML", false, "Flag to apply ML selections"};
  std::shared_ptr<Ort::Experimental::Session> session = nullptr;
  Ort::SessionOptions sessionOptions;
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "ml-model-hf-dplus-selector"};
  std::vector<std::string> inputNamesML{};
  std::vector<std::vector<int64_t>> inputShapesML{};
  std::vector<std::string> outputNamesML{};
  std::vector<std::vector<int64_t>> outputShapesML{};
  std::vector<Ort::Value> inputML = {};
  std::vector<float> dummyOutputML = {};

  /*
  /// Selection on goodness of daughter tracks
  /// \note should be applied at candidate selection
  /// \param track is daughter track
  /// \return true if track is good
  template <typename T>
  bool daughterSelection(const T& track)
  {
    if (track.tpcNClsFound() == 0) {
      return false; //is it clusters findable or found - need to check
    }
    return true;
  }
  */

  /// Candidate selections
  /// \param candidate is candidate
  /// \param trackPion1 is the first track with the pion hypothesis
  /// \param trackKaon is the track with the kaon hypothesis
  /// \param trackPion2 is the second track with the pion hypothesis
  /// \return true if candidate passes all cuts
  template <typename T1, typename T2>
  bool selection(const T1& candidate, const T2& trackPion1, const T2& trackKaon, const T2& trackPion2)
  {
    auto candpT = candidate.pt();
    int pTBin = findBin(binsPt, candpT);
    if (pTBin == -1) {
      return false;
    }
    // check that the candidate pT is within the analysis range
    if (candpT < ptCandMin || candpT > ptCandMax) {
      return false;
    }
    // cut on daughter pT
    if (trackPion1.pt() < cuts->get(pTBin, "pT Pi") || trackKaon.pt() < cuts->get(pTBin, "pT K") || trackPion2.pt() < cuts->get(pTBin, "pT Pi")) {
      return false;
    }
    // invariant-mass cut
    if (std::abs(invMassDplusToPiKPi(candidate) - RecoDecay::getMassPDG(pdg::Code::kDPlus)) > cuts->get(pTBin, "deltaM")) {
      return false;
    }
    if (candidate.decayLength() < cuts->get(pTBin, "decay length")) {
      return false;
    }
    if (candidate.decayLengthXYNormalised() < cuts->get(pTBin, "normalized decay length XY")) {
      return false;
    }
    if (candidate.cpa() < cuts->get(pTBin, "cos pointing angle")) {
      return false;
    }
    if (candidate.cpaXY() < cuts->get(pTBin, "cos pointing angle XY")) {
      return false;
    }
    if (std::abs(candidate.maxNormalisedDeltaIP()) > cuts->get(pTBin, "max normalized deltaIP")) {
      return false;
    }
    return true;
  }

  void init(o2::framework::InitContext&)
  {
    if (b_applyML) {
      std::string onnxFile = std::getenv("MLMODELS_ROOT") + std::string("/models/HF/Tests/XGBoostModel_PbPb3050_pT_12_36_120121_converted.onnx");
      session = std::make_shared<Ort::Experimental::Session>(env, onnxFile, sessionOptions);
      inputNamesML = session->GetInputNames();
      inputShapesML = session->GetInputShapes();
      outputNamesML = session->GetOutputNames();
      outputShapesML = session->GetOutputShapes();
      dummyOutputML.assign(outputShapesML[1][1], 0.f);
      LOG(info) << "Applying ML!";
      LOG(info) << "Number of outputs: " << outputShapesML[1][1];
    }
  }

   void process(aod::HfCand3Prong const& candidates, aod::BigTracksPID const&)
  {
    TrackSelectorPID selectorPion(kPiPlus);
    selectorPion.setRangePtTPC(ptPidTpcMin, ptPidTpcMax);
    selectorPion.setRangeNSigmaTPC(-nSigmaTpcMax, nSigmaTpcMax);
    selectorPion.setRangeNSigmaTPCCondTOF(-nSigmaTpcMax, nSigmaTpcMax);
    selectorPion.setRangePtTOF(ptPidTofMin, ptPidTofMax);
    selectorPion.setRangeNSigmaTOF(-nSigmaTofMax, nSigmaTofMax);
    selectorPion.setRangeNSigmaTOFCondTPC(-nSigmaTofMax, nSigmaTofMax);

    TrackSelectorPID selectorKaon(selectorPion);
    selectorKaon.setPDG(kKPlus);

    // looping over 3-prong candidates
    for (auto& candidate : candidates) {

      // final selection flag:
      auto statusDplusToPiKPi = 0;

      if (!(candidate.hfflag() & 1 << DecayType::DplusToPiKPi)) {
        hfSelDplusToPiKPiCandidate(statusDplusToPiKPi);
        if (b_applyML) {
          hfMlDplusToPiKPiCandidate(dummyOutputML);
        }
        continue;
      }
      SETBIT(statusDplusToPiKPi, aod::SelectionStep::RecoSkims);

      auto trackPos1 = candidate.prong0_as<aod::BigTracksPID>(); // positive daughter (negative for the antiparticles)
      auto trackNeg = candidate.prong1_as<aod::BigTracksPID>();  // negative daughter (positive for the antiparticles)
      auto trackPos2 = candidate.prong2_as<aod::BigTracksPID>(); // positive daughter (negative for the antiparticles)

      /*
      // daughter track validity selection
      if (!daughterSelection(trackPos1) ||
          !daughterSelection(trackNeg) ||
          !daughterSelection(trackPos2)) {
        hfSelDplusToPiKPiCandidate(statusDplusToPiKPi);
        continue;
      }
      */

      // topological selection
      if (!selection(candidate, trackPos1, trackNeg, trackPos2)) {
        hfSelDplusToPiKPiCandidate(statusDplusToPiKPi);
        if (b_applyML) {
          hfMlDplusToPiKPiCandidate(dummyOutputML);
        }
        continue;
      }
      SETBIT(statusDplusToPiKPi, aod::SelectionStep::RecoTopol);

      // track-level PID selection
      int pidTrackPos1Pion = selectorPion.getStatusTrackPIDAll(trackPos1);
      int pidTrackNegKaon = selectorKaon.getStatusTrackPIDAll(trackNeg);
      int pidTrackPos2Pion = selectorPion.getStatusTrackPIDAll(trackPos2);

      if (pidTrackPos1Pion == TrackSelectorPID::Status::PIDRejected ||
          pidTrackNegKaon == TrackSelectorPID::Status::PIDRejected ||
          pidTrackPos2Pion == TrackSelectorPID::Status::PIDRejected) { // exclude D±
        hfSelDplusToPiKPiCandidate(statusDplusToPiKPi);
        if (b_applyML) {
          hfMlDplusToPiKPiCandidate(dummyOutputML);
        }
        continue;
      }
      SETBIT(statusDplusToPiKPi, aod::SelectionStep::RecoPID);

      if (b_applyML) {
        // ML selections
        std::vector<float> inputFeatures{candidate.cpa(), candidate.cpaXY(), candidate.decayLength(), candidate.decayLengthXY(),
                                         candidate.decayLengthXYNormalised(), candidate.impactParameterXY(), 200., 5., 0.8,
                                         candidate.maxNormalisedDeltaIP(), 2., 2., 2., 2., 2., 2.};
        inputML.push_back(Ort::Experimental::Value::CreateTensor<float>(inputFeatures.data(), inputFeatures.size(), inputShapesML[0]));

        auto outputTensor = session->Run(inputNamesML, inputML, outputNamesML);
        auto scores = outputTensor[1].GetTensorMutableData<float>();
        std::vector<float> outputML(scores, scores + outputShapesML[1][1]);
        hfMlDplusToPiKPiCandidate(outputML);

        SETBIT(statusDplusToPiKPi, aod::SelectionStep::RecoMl);

        inputML.clear();
      }

      hfSelDplusToPiKPiCandidate(statusDplusToPiKPi);
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<HfCandidateSelectorDplusToPiKPi>(cfgc)};
}
