'use strict';
const periodic = require('periodicjs');
const mathjs = require('mathjs');
const helpers = require('./helpers');
const { generateProjectedResult, mapPredictionToDigiFiScore, normalize, } = require('../helpers');

function evaluate({ mlmodel, configuration, state, scoreanalysis = null }, /*configuration, state, machinelearning*/) {
  try {
    const sagemakerruntime = periodic.aws.sagemakerruntime;
    const sagemaker_ll = mlmodel[ 'sagemaker_ll' ];
    const datasource = mlmodel.datasource;
    const statistics = datasource.statistics;
    const strategy_data_schema = JSON.parse(datasource.strategy_data_schema);
    const transformations = datasource.transformations;
    const provider_datasource = datasource.providers.sagemaker_ll;
    const model_type = mlmodel.type;
    let done = false;
    let start = new Date();
    let result = {};
    let datasource_headers = provider_datasource.headers.filter(el => el !== 'historical_result');
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
      return column.map(elmt => isNaN(parseFloat(elmt)) ? mean : parseFloat(elmt));
    });
    if (columnTypes[ 'historical_result' ] === 'Boolean' || columnTypes[ 'historical_result' ] === 'Number') {
      cleaned = cleaned.map((column, idx) => {
        let header = datasource_headers[ idx ];
        if (columnTypes[ header ] === 'Number' && transformations[ header ] && transformations[ header ].evaluator) {
          let applyTransformFunc = (columnTypes[ header ] === 'Number') ? new Function('x', transformations[ header ].evaluator) : (elmt) => {
            return elmt; 
          };
          return column.map(applyTransformFunc);
        } else {
          return column;
        }
      });
    }
    cleaned = mathjs.transpose(cleaned)[ 0 ];
    let params = {
      Body: cleaned.join(', '),
      EndpointName: sagemaker_ll.real_time_prediction_id,
      ContentType: 'text/csv',
    };
    let explainability_results = [];
    sagemakerruntime.invokeEndpoint(params, async (err, response) => {
      if (response) {
        response = JSON.parse(Buffer.from(response.Body).toString('utf8'));
        let score;
        let predicted_label;
        let predicted_label_idx;
        if (model_type === 'binary' || model_type === 'regression') {
          score = Number(response.predictions[ 0 ].score);
        } else if (model_type === 'categorical') {
          predicted_label_idx = response.predictions[ 0 ].predicted_label;
          predicted_label_idx = parseInt(predicted_label_idx);
          score = response.predictions[ 0 ].score[ predicted_label_idx ];
          predicted_label = (datasource.decoders && datasource.decoders.historical_result) ? datasource.decoders.historical_result[ predicted_label_idx ] : predicted_label_idx;
        }

        let adr = null;
        let digifi_score = null;
        if (model_type === 'binary') {
          if (scoreanalysis) {
            digifi_score = mapPredictionToDigiFiScore(score);
            adr = generateProjectedResult(scoreanalysis, score);
          }
        }

        await Promise.all(datasource_headers.map(async (header, i) => {
          let new_cleaned = cleaned.slice();
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
          let adjusted_params = {
            Body: new_cleaned.join(', '),
            EndpointName: mlmodel.sagemaker_ll.real_time_prediction_id,
            ContentType: 'text/csv',
          };
          let explainability_result = await sagemakerruntime.invokeEndpoint(adjusted_params).promise();
          explainability_result = JSON.parse(Buffer.from(explainability_result.Body).toString('utf8'));
          if (model_type === 'binary' || model_type === 'regression') {
            explainability_result = Number(explainability_result.predictions[ 0 ].score);
          } else if (model_type === 'categorical') {
            explainability_result = explainability_result.predictions[ 0 ].score[ predicted_label_idx ];
            
          }
          explainability_results.push({ 
            answer: (model_type === 'regression') 
              ? Number((score - explainability_result).toFixed(2))
              : Number((score - explainability_result).toFixed(4)),
            variable: header, 
          })
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
        score = (model_type === 'categorical') ? predicted_label : score;
        const resultsArr = (mlmodel.industry) ? [digifi_score, adr] : [score, ];
        resultsArr.push(...positiveResultsArr, ...negativeResultsArr);
        let mlOutput = configuration.outputs.reduce((aggregate, output, i) => {
          if (output.output_variable) {
            state[output.output_variable] = resultsArr[i];
            aggregate[output.output_variable] = resultsArr[i];
          }
          return aggregate;
        }, {});
        
        result = {
          prediction: response,
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