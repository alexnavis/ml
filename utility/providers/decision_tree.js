'use strict';
const periodic = require('periodicjs');
const mathjs = require('mathjs');
const helpers = require('../helpers');
const Matrix = require('ml-matrix');
const DTClassifier = require('@digifi/ml-cart').DecisionTreeClassifier;
const DTRegression = require('@digifi/ml-cart').DecisionTreeRegression;
const { generateProjectedResult, mapPredictionToDigiFiScore, normalize,  } = require('../helpers');

function evaluate({ mlmodel, configuration, state, scoreanalysis = null }, /*configuration, state, machinelearning*/) {
  try {
    const decision_tree = mlmodel[ 'decision_tree' ];
    const datasource = mlmodel.datasource;
    const statistics = datasource.statistics;
    const ml_input_schema = datasource.included_columns || datasource.strategy_data_schema;
    const strategy_data_schema = JSON.parse(ml_input_schema);
    const transformations = datasource.transformations;
    const provider_datasource = datasource.providers.digifi;
    const model_type = mlmodel.type;
    let done = false;
    let start = new Date();
    let result = {};
    let datasource_headers = provider_datasource.headers.filter(header => header !== 'historical_result');
    let features = datasource_headers.reduce((reduced, model_variable_name) => {
      const variable = configuration.inputs.find((variable) => variable.model_variable_name === model_variable_name);
      if (variable.input_type === 'value') reduced.push((variable.system_variable_id !== undefined && variable.system_variable_id !== null) ? variable.system_variable_id.toString() : null);
      else reduced[ variable.model_variable_name ] = reduced.push((variable.system_variable_id && state[ variable.system_variable_id ] !== undefined && state[ variable.system_variable_id ] !== null) ? state[ variable.system_variable_id ].toString() : null);
      return reduced;
    }, []);
    
    let columnTypes = {};
    for (let [ key, val, ] of Object.entries(strategy_data_schema)) {
      columnTypes[ key ] = val.data_type;
    }
    let transposed_rows = helpers.formatDataTypeColumns({ columnTypes, csv_headers: datasource_headers, rows: [ features ] })
    let hot_encoded = helpers.oneHotEncodeValues({ transposed_rows, columnTypes, encoders: datasource.encoders, decoders: datasource.decoders, encoder_counts: datasource.encoder_counts, csv_headers: datasource_headers, })
    let cleaned = hot_encoded.map((column, idx) => {
      let header = datasource_headers[ idx ];
      let mean = statistics[ header ].mean;
      return column.map(elmt => isNaN(parseFloat(elmt)) ? mean : elmt);
    });
    if (columnTypes[ 'historical_result' ] === 'Boolean' || columnTypes[ 'historical_result' ] === 'Number') {
      cleaned = cleaned.map((column, idx) => {
        let header = datasource_headers[ idx ];
        if (columnTypes[ header ] === 'Number' && transformations[ header ] && transformations[ header ].evaluator) {
          let applyTransformFunc = new Function('x', transformations[ header ].evaluator);
          return column.map(applyTransformFunc);
        } else {
          return column;
        }
      });
    }
    cleaned = mathjs.transpose(cleaned);
    const model_config = JSON.parse(decision_tree.model);
    const classifier = (model_type === 'regression') ? DTRegression.load(model_config) : DTClassifier.load(model_config);
    let prediction = (model_type === 'binary') 
      ? helpers.runDecisionTreePrediction.call(classifier, cleaned) 
      : (model_type === 'categorical') 
        ? helpers.runDecisionTreePredictionCategorical.call(classifier, cleaned) 
        : classifier.predict(cleaned);
    prediction = prediction[ 0 ];
    let adr = null;
    let digifi_score = null;
    let maxValue = null;
    let predictedLabel = null;
    if (model_type === 'binary') {
      if (scoreanalysis) {
        digifi_score = mapPredictionToDigiFiScore(prediction);
        adr = generateProjectedResult(scoreanalysis, prediction);
      }
    } else if (model_type === 'categorical') {
      maxValue = Math.max(...prediction);
      predictedLabel = prediction.indexOf(maxValue);
      prediction = maxValue;
    }

    let explainability_results = [];
    datasource_headers.forEach((header, i) => {
      let new_cleaned = cleaned[ 0 ].slice();
      if (strategy_data_schema[ header ] && strategy_data_schema[ header ].data_type === 'Number') {
        new_cleaned[ i ] = (statistics[ header ] && statistics[ header ].mean !== undefined) ? statistics[ header ].mean : null;
        if (transformations && transformations[ header ] && transformations[ header ].evaluator) {
          let applyTransformFunc = new Function('x', transformations[ header ].evaluator);
          new_cleaned[ i ] = applyTransformFunc(new_cleaned[ i ]);
        }
      } else if (strategy_data_schema[ header ] && (strategy_data_schema[ header ].data_type === 'String' || strategy_data_schema[ header ].data_type === 'Boolean') && statistics[ header ].mode !== undefined) {
        new_cleaned[ i ] = statistics[ header ].mode || null;
        if (new_cleaned[ i ] !== undefined && datasource.encoders[ header ] && datasource.encoders[ header ][ new_cleaned[ i ] ] !== undefined) {
          new_cleaned[ i ] = datasource.encoders[ header ][ new_cleaned[ i ] ];
        } else {
          new_cleaned[ i ] = datasource.encoder_counts[ header ];
        }
      }
      let explainability_result = (model_type === 'binary') 
        ? helpers.runDecisionTreePrediction.call(classifier, [ new_cleaned ]) 
        : (model_type === 'categorical')? helpers.runDecisionTreePredictionCategorical.call(classifier, [new_cleaned]) : classifier.predict([new_cleaned]);

      if (model_type === 'categorical') {
        explainability_result = explainability_result[0][predictedLabel];
      } else {
        explainability_result = explainability_result[0];
      }
      explainability_results.push({ 
        answer: (model_type === 'regression') 
          ? Number((prediction - explainability_result).toFixed(2))
          : Number((prediction - explainability_result).toFixed(4)),
        variable: header, 
      })
    });
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
          positiveResultsArr.push(explainability_results[explainabilityIdx].variable);
        } else {
          positiveResultsArr.push(null);
        }
        positiveCount++;
      }
      if (negativeCount < 5) {
        if (explainability_results[explain_length - explainabilityIdx - 1].answer < 0) {
          negativeResultsArr.push(explainability_results[explain_length - explainabilityIdx - 1].variable);
        } else {
          negativeResultsArr.push(null);
        }
        negativeCount++;
      } 
      explainabilityIdx++;
    }
    prediction = (model_type === 'categorical') ? datasource.decoders.historical_result[predictedLabel] : prediction;
    const resultsArr = (mlmodel.industry) ? [digifi_score, adr] : [prediction,];
    resultsArr.push(...positiveResultsArr, ...negativeResultsArr);
    let mlOutput = configuration.outputs.reduce((aggregate, output, i) => {
      if (output.output_variable) {
        state[output.output_variable] = resultsArr[i];
        aggregate[output.output_variable] = resultsArr[i];
      }
      return aggregate;
    }, {});
    
    result = {
      prediction,
      classification: model_type,
      output: mlOutput,
    };
    return result;
  } catch (e) {
    if (configuration.sync === true) return { err: e, prediction: {}, output: {}, };
    return { err: e, prediction: {}, output: {}, };
  }
}

module.exports = evaluate;