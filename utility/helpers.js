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

function formatDataTypeColumns({ columnTypes, strategy_data_schema, csv_headers, rows, }) {
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

function oneHotEncodeValues({ transposed_rows, columnTypes, encoders, decoders, encoder_counts, csv_headers, }) {
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

function generateProjectedResult(scoreanalysis, prediction) {
  try {
    const evaluatorFunc = new Function('x', scoreanalysis.results.projection_evaluator);
    const projection_adr = evaluatorFunc(mapPredictionToDigiFiScore(prediction));
    if (!isNaN(parseFloat(projection_adr))) return projection_adr < 0 ? 0 : projection_adr;
    else return prediction; 
  } catch(e) {
    return e;
  }
}

function mapPredictionToDigiFiScore(prediction) {
  switch (true) {
  case (prediction < 0.002):
    return 850;
  case (prediction < 0.004):
    return 840;
  case (prediction < 0.006):
    return 830;
  case (prediction < 0.008):
    return 820;
  case (prediction < 0.01):
    return 810;
  case (prediction < 0.015):
    return 800;
  case (prediction < 0.02):
    return 790;
  case (prediction < 0.025):
    return 780;
  case (prediction < 0.03):
    return 770;
  case (prediction < 0.035):
    return 760;
  case (prediction < 0.045):
    return 750;
  case (prediction < 0.055):
    return 740;
  case (prediction < 0.065):
    return 730;
  case (prediction < 0.075):
    return 720;
  case (prediction < 0.085):
    return 710;
  case (prediction < 0.1):
    return 700;
  case (prediction < 0.115):
    return 690;
  case (prediction < 0.13):
    return 680;
  case (prediction < 0.145):
    return 670;
  case (prediction < 0.16):
    return 660;
  case (prediction < 0.175):
    return 650;
  case (prediction < 0.19):
    return 640;
  case (prediction < 0.205):
    return 630;
  case (prediction < 0.22):
    return 620;
  case (prediction < 0.235):
    return 610;
  case (prediction < 0.255):
    return 600;
  case (prediction < 0.275):
    return 590;
  case (prediction < 0.295):
    return 580;
  case (prediction < 0.315):
    return 570;
  case (prediction < 0.335):
    return 560;
  case (prediction < 0.355):
    return 550;
  case (prediction < 0.375):
    return 540;
  case (prediction < 0.395):
    return 530;
  case (prediction < 0.415):
    return 520;
  case (prediction < 0.435):
    return 510;
  case (prediction < 0.46):
    return 500;
  case (prediction < 0.485):
    return 490;
  case (prediction < 0.51):
    return 480;
  case (prediction < 0.535):
    return 470;
  case (prediction < 0.56):
    return 460;
  case (prediction < 0.585):
    return 450;
  case (prediction < 0.61):
    return 440;
  case (prediction < 0.635):
    return 430;
  case (prediction < 0.66):
    return 420;
  case (prediction < 0.685):
    return 410;
  case (prediction < 0.715):
    return 300;
  case (prediction < 0.745):
    return 390;
  case (prediction < 0.775):
    return 380;
  case (prediction < 0.805):
    return 370;
  case (prediction < 0.835):
    return 360;
  case (prediction < 0.865):
    return 350;
  case (prediction < 0.895):
    return 340;
  case (prediction < 0.925):
    return 330;
  case (prediction < 0.955):
    return 320;
  case (prediction < 0.985):
    return 310;
  case (prediction <= 1):
    return 300;
  default:
    return 300;
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
  generateProjectedResult, 
  mapPredictionToDigiFiScore, 
  normalize, 
  formatDataTypeColumns,
  oneHotEncodeValues,
  runDecisionTreePrediction,
  runDecisionTreePredictionCategorical,
  runRandomForestPrediction,
  runRandomForestPredictionCategorical,
};