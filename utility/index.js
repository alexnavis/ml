'use strict';
const periodic = require('periodicjs');
const PROVIDER_EVALUATORS = require('./providers');

function evaluate(configuration, state, machinelearning) {
  try {
    if (configuration) {
      let done = false;
      let start = new Date();
      let result = {};
      const MLModel = periodic.datas.get('standard_mlmodel');
      const ScoreAnalysis = periodic.datas.get('standard_scoreanalysis');
      MLModel.load({ query: { _id: configuration.mlmodel_id } })
        .then(async mlmodel => {
          mlmodel = mlmodel.toJSON ? mlmodel.toJSON() : mlmodel;
          let scoreanalysis = null;
          if (mlmodel.industry) scoreanalysis = await ScoreAnalysis.model.findOne({ mlmodel: mlmodel._id.toString(), type: 'training', provider: mlmodel.selected_provider }).lean();
          let provider = mlmodel.selected_provider;
          result = await PROVIDER_EVALUATORS[ provider ]({ mlmodel, configuration, state, scoreanalysis });
          done = true;
        })
        .catch(e => {
          done = true;
        });
      require('deasync').loopWhile(() => {
        if (new Date() - start > 3000) return false;
        return !done;
      });
      return result;
    } else {
      return { err: { message: 'Output Variable is Required.' }, prediction: {}, output: {}, }
    }
  } catch (e) {
    if (configuration.sync === true) return { err: e, prediction: {}, output: {}, };
    return { err: e, prediction: {}, output: {}, };
  }
}

module.exports = {
  evaluate,
};