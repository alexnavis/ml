'use strict';
const periodic = require('periodicjs');
const mathjs = require('mathjs');
const helpers = require('./helpers');
const { generateProjectedResult, mapPredictionToDigiFiScore, normalize,  } = require('../helpers');

function evaluate({ mlmodel, configuration, state, scoreanalysis = null }, /*configuration, state, machinelearning*/) {
  try {
    const neural_network = mlmodel[ 'neural_network' ];
    const datasource = mlmodel.datasource;
    const statistics = datasource.statistics;
    const ml_input_schema = datasource.included_columns || datasource.strategy_data_schema;
    const strategy_data_schema = JSON.parse(ml_input_schema);
    const transformations = datasource.transformations;
    const provider_datasource = datasource.providers.digifi;
    const column_scale = neural_network.column_scale;
    const model_type = mlmodel.type;
    let done = false;
    let start = new Date();
    let result = {};
    let datasource_headers = provider_datasource.headers.filter(header => header !== 'historical_result');
    let ml_variables = {};
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
    let transposed_rows = helpers.formatDataTypeColumns({ columnTypes, csv_headers: datasource_headers, rows: [ features, ], });
    let hot_encoded = helpers.oneHotEncodeValues({ transposed_rows, columnTypes, encoders: datasource.encoders, decoders: datasource.decoders, encoder_counts: datasource.encoder_counts, csv_headers: datasource_headers, });
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
    cleaned = mathjs.transpose(cleaned)[ 0 ];
    cleaned = cleaned.map((col, idx) => {
      let { min, max, } = column_scale[ datasource_headers[ idx ] ];
      col = helpers.normalize(min, max)(col);
      return col;
    });
    const model_config = neural_network.model;
    let classifier;
    eval(`classifier= ${model_config}`);
    let prediction = classifier(cleaned);
    let adr = null;
    let digifi_score = null;
    let maxValue = null;
    let predictedLabel = null;
    if (model_type === 'binary') {
      if (scoreanalysis) {
        prediction = prediction['true'];
        digifi_score = mapPredictionToDigiFiScore(prediction['true']);
        adr = generateProjectedResult(scoreanalysis, prediction[ 'true' ]);
      } else {
        prediction = prediction[ 'true' ];
      }
    } else if (model_type === 'categorical') {
      prediction = Object.keys(prediction).reduce((arr, key) => {
        arr[ Number(key) ] = prediction[ key ];
        return arr;
      }, []);
      maxValue = Math.max(...prediction);
      predictedLabel = prediction.indexOf(maxValue);
      prediction = maxValue;
    }

    let explainability_results = [];
    datasource_headers.forEach((header, i) => {
      let new_cleaned = cleaned.slice();
      if (strategy_data_schema[ header ] && strategy_data_schema[ header ].data_type === 'Number') {
        new_cleaned[ i ] = (statistics[ header ] && statistics[ header ].mean !== undefined) ? statistics[ header ].mean : null;
        if (transformations && transformations[ header ] && transformations[ header ].evaluator) {
          let applyTransformFunc = new Function('x', transformations[ header ].evaluator);
          new_cleaned[ i ] = applyTransformFunc(new_cleaned[ i ]);
        }
        let { min, max, } = column_scale[ datasource_headers[ i ] ];
        new_cleaned[ i ] = normalize(min, max)(new_cleaned[ i ]);
      } else if (strategy_data_schema[ header ] && (strategy_data_schema[ header ].data_type === 'String' || strategy_data_schema[ header ].data_type === 'Boolean') && statistics[ header ].mode !== undefined) {
        new_cleaned[ i ] = statistics[ header ].mode || null;
        if (new_cleaned[ i ] !== undefined && datasource.encoders[ header ] && datasource.encoders[ header ][ new_cleaned[ i ] ] !== undefined) {
          new_cleaned[ i ] = datasource.encoders[ header ][ new_cleaned[ i ] ];
        } else {
          new_cleaned[ i ] = datasource.encoder_counts[ header ];
        }
        let { min, max, } = column_scale[ datasource_headers[ i ] ];
        new_cleaned[ i ] = normalize(min, max)(new_cleaned[ i ]);
      }
      let explainability_result = classifier(new_cleaned);
      if (mlmodel.type === 'binary') {
        explainability_result = !isNaN(parseFloat(explainability_result[ 'true' ])) ? explainability_result[ 'true' ] : 1 - explainability_result[ 'false' ];
      } else if (mlmodel.type === 'categorical') {
        explainability_result = explainability_result[predictedLabel];
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
    const resultsArr = (mlmodel.industry) ? [digifi_score, adr] : [prediction];
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
      classification: mlmodel.type,
      output: mlOutput,
    };
    return result;
  } catch (e) {
    if (configuration.sync === true) return { err: e, prediction: {}, output: {}, };
    return { err: e, prediction: {}, output: {}, };
  }
}

module.exports = evaluate;