'use strict';
const mathjs = require('mathjs');
const Matrix = require('ml-matrix');

function runDecisionTreePrediction(dataset) {
  let toPredict = Matrix.Matrix.checkMatrix(dataset);
  var predictions = new Array(toPredict.rows);
  for (var i = 0; i < toPredict.rows; ++i) {
    predictions[ i ] = this.root.classify(toPredict.getRow(i))[ 0 ][ 1 ] || 0;
  }
  return predictions;
}

function runDecisionTreePredictionCategorical(dataset) {
  let toPredict = Matrix.Matrix.checkMatrix(dataset);
  var predictions = new Array(toPredict.rows);
  for (var i = 0; i < toPredict.rows; ++i) {
    predictions[ i ] = this.root.classify(toPredict.getRow(i))[ 0 ];
  }
  return predictions;
}

function runRandomForestPrediction(toPredict) {
  try {
    let predictionValues = new Array(this.nEstimators);
    toPredict = Matrix.Matrix.checkMatrix(toPredict);
    for (var i = 0; i < this.nEstimators; ++i) {
      let X = toPredict.columnSelectionView(this.indexes[ i ]);
      predictionValues[ i ] = runDecisionTreePrediction.call(this.estimators[ i ], X);
    }
    predictionValues = new Matrix.WrapperMatrix2D(predictionValues).transposeView();
    let predictions = new Array(predictionValues.rows);
    for (i = 0; i < predictionValues.rows; ++i) {
      predictions[ i ] = this.selection(predictionValues.getRow(i));
    }
    return predictions;
  } catch (e) {
    return e;
  }
}

function runRandomForestPredictionCategorical(toPredict) {
  try {
    let predictionValues = new Array(this.nEstimators);
    toPredict = Matrix.Matrix.checkMatrix(toPredict);
    for (var i = 0; i < this.nEstimators; ++i) {
      let X = toPredict.columnSelectionView(this.indexes[ i ]);
      predictionValues[ i ] = runDecisionTreePredictionCategorical.call(this.estimators[ i ], X);
    }
    predictionValues = new Matrix.WrapperMatrix2D(predictionValues).transposeView();
    let predictions = new Array(predictionValues.rows);
    for (i = 0; i < predictionValues.rows; ++i) {
      predictions[ i ] = this.selection(predictionValues.getRow(i));
    }
    return predictions;
  } catch (e) {
    return e;
  }
}

function formatDataTypeColumns({ columnTypes, strategy_data_schema, csv_headers, rows }) {
  try {
    let transposedrows = mathjs.transpose(rows);
    csv_headers.forEach((header, idx) => {
      if (columnTypes[ header ] === 'Date') {
        transposedrows[ idx ] = transposedrows[ idx ].map(celldata => new Date(celldata).getTime());
      }
    });
    return transposedrows;
  } catch (e) {
    return e;
  }
}

function oneHotEncodeValues({ transposed_rows, columnTypes, encoders, decoders, encoder_counts, csv_headers }) {
  try {
    let hot_encoded_rows = transposed_rows.map((column, idx) => {
      let header = csv_headers[ idx ];
      if (columnTypes[ header ] === 'String' || columnTypes[ header ] === 'Boolean') {
        return column.map(data => {
          if (!isNaN(encoders[ header ][ data ])) return encoders[ header ][ data ];
          else return encoder_counts[ header ];
        });
      } else {
        return column;
      }
    });
    return hot_encoded_rows;
  } catch (e) {
    return e;
  }
}

function normalize(min, max) {
  const delta = max - min;
  return function (val) {
    let scaled = (val - min) / delta;
    if (scaled > 1) scaled = 1;
    if (scaled < 0) scaled = 0;
    return scaled;
  };
}

module.exports = {
  formatDataTypeColumns,
  oneHotEncodeValues,
  normalize,
  runDecisionTreePrediction,
  runDecisionTreePredictionCategorical,
  runRandomForestPrediction,
  runRandomForestPredictionCategorical,
}