'use strict';
const periodic = require('periodicjs');
const mathjs = require('mathjs');
const helpers = require('./helpers');
const { generateProjectedResult, mapPredictionToDigiFiScore,  } = require('../helpers');

function evaluate({ mlmodel, configuration, state, scoreanalysis = null }, /*configuration, state, machinelearning*/) {
  try {
    let explainability_results = [];
    const machinelearning = periodic.aws.machinelearning;
    let providerdata = mlmodel[ 'aws' ];
    const model_type = mlmodel.type;
    const datasource = mlmodel.datasource;
    const strategy_data_schema = JSON.parse(datasource.strategy_data_schema);
    let ml_statistics = datasource.statistics;
    let aws_data_schema_map = JSON.parse(mlmodel.datasource.data_schema).attributes.reduce((aggregate, config, i) => {
      aggregate[ config.attributeName ] = config.attributeType;
      return aggregate;
    }, {});
    const transformations = datasource.transformations;
    let columnTypes = {};
    for (let [ key, val, ] of Object.entries(strategy_data_schema)) {
      columnTypes[ key ] = val.data_type;
    }

    let ml_variables = configuration.inputs.reduce((reduced, variable) => {
      if (variable.input_type === 'value') reduced[ variable.model_variable_name ] = (variable.system_variable_id !== undefined && variable.system_variable_id !== null) ? variable.system_variable_id.toString() : null;
      else reduced[ variable.model_variable_name ] = (variable.system_variable_id && state[ variable.system_variable_id ] !== undefined && state[ variable.system_variable_id ] !== null) ? state[ variable.system_variable_id ].toString() : null;
      if (reduced[ variable.model_variable_name ] && columnTypes[ variable.model_variable_name ] && columnTypes[ variable.model_variable_name ] === 'Date') {
        reduced[ variable.model_variable_name ] = new Date(reduced[ variable.model_variable_name ]).getTime().toString();
      }
      return reduced;
    }, {});
    let params = {
      MLModelId: providerdata.real_time_prediction_id,
      PredictEndpoint: providerdata.real_time_endpoint,
      Record: ml_variables,
    };
    let done = false;
    let start = new Date();
    let result = {};
    machinelearning.predict(params, async function (err, prediction) {
      if (err) {
        result = { err, prediction: {}, output: {}, }
      } else {
        let score;
        let categoricalResult;
        if (prediction[ 'Prediction' ][ 'details' ][ 'PredictiveModelType' ] === 'BINARY') {
          score = prediction[ 'Prediction' ][ 'predictedScores' ][ prediction[ 'Prediction' ][ 'predictedLabel' ] ];
        } else if (prediction[ 'Prediction' ][ 'details' ][ 'PredictiveModelType' ] === 'REGRESSION') {
          score = prediction[ 'Prediction' ][ 'predictedValue' ];
        } else if (prediction[ 'Prediction' ][ 'details' ][ 'PredictiveModelType' ] === 'MULTICLASS') {
          score = prediction[ 'Prediction' ][ 'predictedScores' ][ prediction[ 'Prediction' ][ 'predictedLabel' ] ];
          categoricalResult = prediction[ 'Prediction' ][ 'predictedLabel' ];
        }
        let digifi_score = null;
        let adr = null;
        if (mlmodel.type === 'binary' && scoreanalysis) {
          digifi_score = mapPredictionToDigiFiScore(score);
          adr = generateProjectedResult(scoreanalysis, score);
        }
        await Promise.all(Object.keys(ml_variables).map(async (variable) => {
          let new_variable_value = null;
          if (strategy_data_schema[ variable ] && strategy_data_schema[ variable ].data_type === 'Number') {
            if (aws_data_schema_map[ variable ] === 'CATEGORICAL') new_variable_value = (ml_statistics[ variable ] && ml_statistics[ variable ].mode !== undefined) ? String(ml_statistics[ variable ].mode) : null;
            else new_variable_value = (ml_statistics[ variable ] && ml_statistics[ variable ].mean !== undefined) ? String(ml_statistics[ variable ].mean) : null;
          } else if (strategy_data_schema[ variable ] && (strategy_data_schema[ variable ].data_type === 'String' || strategy_data_schema[ variable ].data_type === 'Boolean') && ml_statistics[ variable ].mode !== undefined) new_variable_value = ml_statistics[ variable ].mode || null;
          let adjusted_params = Object.assign({}, params, {
            Record: Object.assign({}, ml_variables, {
              [ `${variable}` ]: new_variable_value,
            })
          })
          let new_prediction = await machinelearning.predict(adjusted_params).promise();
          let answer;
          if (new_prediction[ 'Prediction' ][ 'details' ][ 'PredictiveModelType' ] === 'BINARY') {
            answer = new_prediction[ 'Prediction' ][ 'predictedScores' ][ new_prediction[ 'Prediction' ][ 'predictedLabel' ] ];
          } else if (new_prediction[ 'Prediction' ][ 'details' ][ 'PredictiveModelType' ] === 'REGRESSION') {
            answer = new_prediction[ 'Prediction' ][ 'predictedValue' ];
          } else if (new_prediction[ 'Prediction' ][ 'details' ][ 'PredictiveModelType' ] === 'MULTICLASS') {
            answer = new_prediction[ 'Prediction' ][ 'predictedScores' ][ new_prediction[ 'Prediction' ][ 'predictedLabel' ] ];
          }
          explainability_results.push({  answer: (model_type === 'regression') ? Number((Number(score) - Number(answer)).toFixed(2)) : Number((Number(score) - Number(answer)).toFixed(4)), variable });
        }));

        explainability_results.sort((a, b) => b.answer - a.answer);
        let positiveCount = 0;
        let negativeCount = 0;
        let explainabilityIdx = 0;
        let positiveResultsArr = [];
        let negativeResultsArr = [];
        const explain_length = explainability_results.length;
        while (positiveCount < 5 && negativeCount < 5) {
          if (positiveCount < 5) {
            if (explainabilityIdx < explain_length && explainability_results[explainabilityIdx].answer > 0) {
              positiveResultsArr.push(explainability_results[explainabilityIdx].variable)
            } else {
              positiveResultsArr.push(null);
            }
            positiveCount++;
          }
          if (negativeCount < 5) {
            if (explainability_results[explain_length - explainabilityIdx - 1].answer < 0) {
              negativeResultsArr.push(explainability_results[explain_length - explainabilityIdx - 1].variable)
            } else {
              negativeResultsArr.push(null);
            }
            negativeCount++;
          } 
          explainabilityIdx++;
        }
        score = (model_type === 'categorical') ? categoricalResult : score;
        const resultsArr = (mlmodel.industry) ? [digifi_score, adr] : [score];
        resultsArr.push(...positiveResultsArr, ...negativeResultsArr);
        let mlOutput = configuration.outputs.reduce((aggregate, output, i) => {
          if (output.output_variable) {
            state[output.output_variable] = resultsArr[i];
            aggregate[output.output_variable] = resultsArr[i];
          }
          return aggregate;
        }, {})

        result = {
          prediction,
          classification: model_type,
          output: mlOutput,
        };
      }
      done = true;
    });
    require('deasync').loopWhile(() => {
      if (new Date() - start > 3000) return false;
      return !done;
    });
    return result;
  } catch (e) {
    if (configuration.sync === true) return { err: e, prediction: {}, output: {}, };
    return { err: e, prediction: {}, output: {}, };
  }
}

module.exports = evaluate;