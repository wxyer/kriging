/*
eg.
var variogram = K.kriging.train(values, lngs, lats, model, sigma2, alpha);
var grid = K.kriging.grid(_polygons, variogram, width);
K.kriging.plot(this.canvas, grid, [extent.xmin, extent.xmax], [extent.ymin, extent.ymax], colors);
*/

export enum kernelModel {
  gaussian = 'gaussian',
  exponential = 'exponential',
  spherical = 'spherical',
}

export class Grid extends Array {
  public width: number;
  public zlim: [number, number];
  public xlim: [number, number];
  public ylim: [number, number];
}

export class Variogram {
  xValues: number[] = [];
  yValues: number[] = [];
  zValues: number[] = [];
  // tslint:disable-next-line: no-inferrable-types
  nugget: number = 0.0;
  // tslint:disable-next-line: no-inferrable-types
  range: number = 0.0;
  // tslint:disable-next-line: no-inferrable-types
  sill: number = 0.0;
  A: number = 1 / 3;
  K: number[] = [];
  M: number[] = [];
  // tslint:disable-next-line: no-inferrable-types
  N: number = 0;
  model: (h: number, nugget: number, range: number, sill: number, A: number) => number;
}

export class Kriging {

  public kernelModel: kernelModel;
  public variogram = new Variogram();
  public colors: string[] = ['#006837', '#1a9850', '#66bd63', '#a6d96a', '#d9ef8b', '#ffffbf',
  '#fee08b', '#fdae61', '#f46d43', '#d73027', '#a50026'];

  constructor(xvalues: number[], yvalues: number[], zvalues: number[], model: kernelModel, ) {
    this.variogram = {
      xValues: xvalues,
      yValues: yvalues,
      zValues: zvalues,
      nugget: 0.0,
      range: 0.0,
      sill: 0.0,
      A: 1 / 3,
      N: 0,
      K: [],
      M: [],
      model: null,
    };
    switch (model) {
      case kernelModel.gaussian:
        this.variogram.model = this.kriging_variogram_gaussian;
        break;
      case kernelModel.exponential:
        this.variogram.model = this.kriging_variogram_exponential;
        break;
      case kernelModel.spherical:
        this.variogram.model = this.kriging_variogram_spherical;
        break;
    }
  }

  // Train using gaussian processes with bayesian priors
  public train(sigma2: number = 0, alpha: number = 100) {

    // Lag distance/semivariance
    let n = this.variogram.zValues.length;
    const distance = Array((n * n - n) / 2);
    for (let i = 0, k = 0; i < n; i++) {
      for (let j = 0; j < i; j++ , k++) {
        distance[k] = Array(2);
        distance[k][0] = Math.pow(
          Math.pow(this.variogram.xValues[i] - this.variogram.xValues[j], 2) +
          Math.pow(this.variogram.yValues[i] - this.variogram.yValues[j], 2), 0.5);
        distance[k][1] = Math.abs(this.variogram.zValues[i] - this.variogram.zValues[j]);
      }
    }
    distance.sort((a, b) => a[0] - b[0]);
    this.variogram.range = distance[(n * n - n) / 2 - 1][0];

    // Bin lag distance
    const lags = ((n * n - n) / 2) > 30 ? 30 : (n * n - n) / 2;
    const tolerance = this.variogram.range / lags;
    const lag = [0].rep(lags);
    const semi = [0].rep(lags);
    let l = 0;
    if (lags < 30) {
      for (l = 0; l < lags; l++) {
        lag[l] = distance[l][0];
        semi[l] = distance[l][1];
      }
    } else {
      for (let i = 0, j = 0, k = 0; i < lags && j < ((n * n - n) / 2); i++ , k = 0) {
        while (distance[j][0] <= ((i + 1) * tolerance)) {
          lag[l] += distance[j][0];
          semi[l] += distance[j][1];
          j++; k++;
          if (j >= ((n * n - n) / 2)) { break; }
        }
        if (k > 0) {
          lag[l] /= k;
          semi[l] /= k;
          l++;
        }
      }
      if (l < 2) { return this.variogram; } // Error: Not enough points
    }

    // Feature transformation
    n = l;
    this.variogram.range = lag[n - 1] - lag[0];
    const X = [1].rep(2 * n);
    const Y = Array(n);
    const A = this.variogram.A;
    for (let i = 0; i < n; i++) {
      switch (this.kernelModel) {
        case kernelModel.gaussian:
          X[i * 2 + 1] = 1.0 - Math.exp(-(1.0 / A) * Math.pow(lag[i] / this.variogram.range, 2));
          break;
        case kernelModel.exponential:
          X[i * 2 + 1] = 1.0 - Math.exp(-(1.0 / A) * lag[i] / this.variogram.range);
          break;
        case kernelModel.spherical:
          X[i * 2 + 1] = 1.5 * (lag[i] / this.variogram.range) -
            0.5 * Math.pow(lag[i] / this.variogram.range, 3);
          break;
      }
      Y[i] = semi[i];
    }

    // Least squares
    const Xt = this.kriging_matrix_transpose(X, n, 2);
    let Z = this.kriging_matrix_multiply(Xt, X, 2, n, 2);
    Z = this.kriging_matrix_add(Z, this.kriging_matrix_diag(1 / alpha, 2), 2, 2);
    const cloneZ = Z.slice(0);
    if (this.kriging_matrix_chol(Z, 2)) {
      this.kriging_matrix_chol2inv(Z, 2);
    } else {
      this.kriging_matrix_solve(cloneZ, 2);
      Z = cloneZ;
    }
    const W = this.kriging_matrix_multiply(this.kriging_matrix_multiply(Z, Xt, 2, 2, n), Y, 2, n, 1);

    // Variogram parameters
    this.variogram.nugget = W[0];
    this.variogram.sill = W[1] * this.variogram.range + this.variogram.nugget;
    this.variogram.N = this.variogram.xValues.length;

    // Gram matrix with prior
    n = this.variogram.xValues.length;
    const K = Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < i; j++) {
        K[i * n + j] = this.variogram.model(Math.pow(Math.pow(this.variogram.xValues[i] - this.variogram.xValues[j], 2) +
          Math.pow(this.variogram.yValues[i] - this.variogram.yValues[j], 2), 0.5),
          this.variogram.nugget,
          this.variogram.range,
          this.variogram.sill,
          this.variogram.A);
        K[j * n + i] = K[i * n + j];
      }
      K[i * n + i] = this.variogram.model(0, this.variogram.nugget,
        this.variogram.range,
        this.variogram.sill,
        this.variogram.A);
    }

    // Inverse penalized Gram matrix projected to target vector
    let C = this.kriging_matrix_add(K, this.kriging_matrix_diag(sigma2, n), n, n);
    const cloneC = C.slice(0);
    if (this.kriging_matrix_chol(C, n)) {
      this.kriging_matrix_chol2inv(C, n);
    } else {
      this.kriging_matrix_solve(cloneC, n);
      C = cloneC;
    }

    // Copy unprojected inverted matrix as K
    const K_C = C.slice(0);
    const M = this.kriging_matrix_multiply(C, this.variogram.zValues, n, n, 1);
    this.variogram.K = K_C;
    this.variogram.M = M;

    return this.variogram;
  }

  // Model prediction
  public predict(x: number, y: number): number {
    const k = Array(this.variogram.N);
    for (let i = 0; i < this.variogram.N; i++) {
      k[i] = this.variogram.model(Math.pow(Math.pow(x - this.variogram.xValues[i], 2) +
        Math.pow(y - this.variogram.yValues[i], 2), 0.5),
        this.variogram.nugget, this.variogram.range,
        this.variogram.sill, this.variogram.A);
    }
    return this.kriging_matrix_multiply(k, this.variogram.M, 1, this.variogram.N, 1)[0];
  }

  public variance(x: number, y: number): number {
    const k = Array(this.variogram.N);
    for (let i = 0; i < this.variogram.N; i++) {
      k[i] = this.variogram.model(Math.pow(Math.pow(x - this.variogram.xValues[i], 2) +
        Math.pow(y - this.variogram.yValues[i], 2), 0.5),
        this.variogram.nugget, this.variogram.range,
        this.variogram.sill, this.variogram.A);
    }
    return this.variogram.model(0, this.variogram.nugget, this.variogram.range,
      this.variogram.sill, this.variogram.A) +
      this.kriging_matrix_multiply(this.kriging_matrix_multiply(k, this.variogram.K,
        1, this.variogram.N, this.variogram.N),
        k, 1, this.variogram.N, 1)[0];
  }

  // Gridded matrices or contour paths
  // OpenLayers鐨刾olygon锛歰l.geom.Polygon(coordinates, opt_layout)
  public grid(polygons: number[][][], gridsize: number): Grid {
    const n = polygons.length;
    if (n === 0) { return; }

    // Boundaries of polygons space
    const xlim: [number, number] = [polygons[0][0][0], polygons[0][0][0]];
    const ylim: [number, number] = [polygons[0][0][1], polygons[0][0][1]];
    for (let i = 0; i < n; i++) { // Polygons
      // tslint:disable-next-line: prefer-for-of
      for (let j = 0; j < polygons[i].length; j++) { // Vertices
        if (polygons[i][j][0] < xlim[0]) {
          xlim[0] = polygons[i][j][0];
        }
        if (polygons[i][j][0] > xlim[1]) {
          xlim[1] = polygons[i][j][0];
        }
        if (polygons[i][j][1] < ylim[0]) {
          ylim[0] = polygons[i][j][1];
        }
        if (polygons[i][j][1] > ylim[1]) {
          ylim[1] = polygons[i][j][1];
        }
      }
    }
    // Alloc for O(n^2) space
    let xtarget: number;
    let ytarget: number;
    const a = Array(2);
    const b = Array(2);
    const lxlim = Array(2); // Local dimensions
    const lylim = Array(2); // Local dimensions
    const x = Math.ceil((xlim[1] - xlim[0]) / gridsize); // x鏂瑰悜涓婄殑鏍煎瓙鏁�
    const y = Math.ceil((ylim[1] - ylim[0]) / gridsize); // y鏂瑰悜涓婄殑鏍煎瓙鏁�

    const gridParam = new Grid(x + 1);
    for (let i = 0; i <= x; i++) { gridParam[i] = Array(y + 1); }// A鏄竴涓簩缁寸煩闃�
    for (let i = 0; i < n; i++) {
      // Range for polygons[i]
      lxlim[0] = polygons[i][0][0];
      lxlim[1] = lxlim[0];
      lylim[0] = polygons[i][0][1];
      lylim[1] = lylim[0];
      for (let j = 1; j < polygons[i].length; j++) { // Vertices
        if (polygons[i][j][0] < lxlim[0]) {
          lxlim[0] = polygons[i][j][0];
        }
        if (polygons[i][j][0] > lxlim[1]) {
          lxlim[1] = polygons[i][j][0];
        }
        if (polygons[i][j][1] < lylim[0]) {
          lylim[0] = polygons[i][j][1];
        }
        if (polygons[i][j][1] > lylim[1]) {
          lylim[1] = polygons[i][j][1];
        }
      }

      // Loop through polygon subspace
      a[0] = Math.floor(((lxlim[0] - ((lxlim[0] - xlim[0]) % gridsize)) - xlim[0]) / gridsize);
      a[1] = Math.ceil(((lxlim[1] - ((lxlim[1] - xlim[1]) % gridsize)) - xlim[0]) / gridsize);
      b[0] = Math.floor(((lylim[0] - ((lylim[0] - ylim[0]) % gridsize)) - ylim[0]) / gridsize);
      b[1] = Math.ceil(((lylim[1] - ((lylim[1] - ylim[1]) % gridsize)) - ylim[0]) / gridsize);
      for (let j = a[0]; j <= a[1]; j++) {
        for (let k = b[0]; k <= b[1]; k++) {
          xtarget = xlim[0] + j * gridsize;
          ytarget = ylim[0] + k * gridsize;
          if (polygons[i].pip(xtarget, ytarget)) {
            gridParam[j][k] = this.predict(xtarget, ytarget);
          }
        }
      }
    }
    gridParam.xlim = xlim;
    gridParam.ylim = ylim;
    gridParam.zlim = [this.variogram.zValues.min(), this.variogram.zValues.max()];
    gridParam.width = gridsize;
    return gridParam;
  }
  // 未实现
  public contour(value: number[], polygons: number[][][]) {

  }
  // Plotting on the DOM
  public plot(canvas: HTMLCanvasElement, grid: Grid, xlim: [number, number], ylim: [number, number], colors: string[]= this.colors) {
    // Clear screen
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Starting boundaries
    const range = [xlim[1] - xlim[0], ylim[1] - ylim[0], grid.zlim[1] - grid.zlim[0]];

    const n = grid.length;
    const m = grid[0].length;
    const wx = Math.ceil(grid.width * canvas.width / (xlim[1] - xlim[0]));
    const wy = Math.ceil(grid.width * canvas.height / (ylim[1] - ylim[0]));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < m; j++) {
        if (grid[i][j] === undefined) { continue; }
        const x = canvas.width * (i * grid.width + grid.xlim[0] - xlim[0]) / range[0];
        const y = canvas.height * (1 - (j * grid.width + grid.ylim[0] - ylim[0]) / range[1]);
        let z = (grid[i][j] - grid.zlim[0]) / range[2];
        if (z < 0.0) { z = 0.0; }
        if (z > 1.0) { z = 1.0; }

        ctx.fillStyle = colors[Math.floor((colors.length - 1) * z)];
        ctx.fillRect(Math.round(x - wx / 2), Math.round(y - wy / 2), wx, wy);
      }
    }
  }

  public plotRainball(canvas: HTMLCanvasElement, grid: Grid, xlim: [number, number],
                      ylim: [number, number], rainbowFunc: (z: number) => string) {
    // Clear screen
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Starting boundaries
    const range = [xlim[1] - xlim[0], ylim[1] - ylim[0], grid.zlim[1] - grid.zlim[0]];
    const n = grid.length;
    const m = grid[0].length;
    const wx = Math.ceil(grid.width * canvas.width / (xlim[1] - xlim[0]));
    const wy = Math.ceil(grid.width * canvas.height / (ylim[1] - ylim[0]));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < m; j++) {
        if (grid[i][j] === undefined) { continue; }
        const x = canvas.width * (i * grid.width + grid.xlim[0] - xlim[0]) / range[0];
        const y = canvas.height * (1 - (j * grid.width + grid.ylim[0] - ylim[0]) / range[1]);
        let z = (grid[i][j] - grid.zlim[0]) / range[2];
        if (z < 0.0) { z = 0.0; }
        if (z > 1.0) { z = 1.0; }

        ctx.fillStyle = '#' + rainbowFunc(z);
        ctx.fillRect(Math.round(x - wx / 2), Math.round(y - wy / 2), wx, wy);
      }
    }
  }


  // Matrix algebra
  private kriging_matrix_diag(c: number, n: number) {
    const Z = [0].rep(n * n);
    for (let i = 0; i < n; i++) {
      Z[i * n + i] = c;
    }
    return Z;
  }
  private kriging_matrix_transpose(X, n, m) {
    const Z = Array(m * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < m; j++) {
        Z[j * n + i] = X[i * m + j];
      }
    }
    return Z;
  }
  private kriging_matrix_scale(X, c, n, m) {
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < m; j++) {
        X[i * m + j] *= c;
      }
    }
  }
  private kriging_matrix_add(X, Y, n, m) {
    const Z = Array(n * m);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < m; j++) {
        Z[i * m + j] = X[i * m + j] + Y[i * m + j];
      }
    }
    return Z;
  }
  // Naive matrix multiplication
  private kriging_matrix_multiply(X, Y, n, m, p) {
    const Z = Array(n * p);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < p; j++) {
        Z[i * p + j] = 0;
        for (let k = 0; k < m; k++) {
          Z[i * p + j] += X[i * m + k] * Y[k * p + j];
        }
      }
    }
    return Z;
  }
  // Cholesky decomposition
  private kriging_matrix_chol(X, n) {
    const p = Array(n);
    for (let i = 0; i < n; i++) { p[i] = X[i * n + i]; }
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < i; j++) {
        p[i] -= X[i * n + j] * X[i * n + j];
      }
      if (p[i] <= 0) { return false; }
      p[i] = Math.sqrt(p[i]);
      for (let j = i + 1; j < n; j++) {
        for (let k = 0; k < i; k++) {
          X[j * n + i] -= X[j * n + k] * X[i * n + k];
        }
        X[j * n + i] /= p[i];
      }
    }
    for (let i = 0; i < n; i++) { X[i * n + i] = p[i]; }
    return true;
  }
  // Inversion of cholesky decomposition
  private kriging_matrix_chol2inv(X, n) {
    let sum = 0;
    for (let i = 0; i < n; i++) {
      X[i * n + i] = 1 / X[i * n + i];
      for (let j = i + 1; j < n; j++) {
        sum = 0;
        for (let k = i; k < j; k++) {
          sum -= X[j * n + k] * X[k * n + i];
        }
        X[j * n + i] = sum / X[j * n + j];
      }
    }
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        X[i * n + j] = 0;
      }
    }
    for (let i = 0; i < n; i++) {
      X[i * n + i] *= X[i * n + i];
      for (let k = i + 1; k < n; k++) {
        X[i * n + i] += X[k * n + i] * X[k * n + i];
      }
      for (let j = i + 1; j < n; j++) {
        for (let k = j; k < n; k++) {
          X[i * n + j] += X[k * n + i] * X[k * n + j];
        }
      }
    }
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < i; j++) {
        X[i * n + j] = X[j * n + i];
      }
    }

  }
  // Inversion via gauss-jordan elimination
  private kriging_matrix_solve(X, n) {
    const m = n;
    const b = Array(n * n);
    const indxc = Array(n);
    const indxr = Array(n);
    const ipiv = Array(n);
    let icol = 0;
    let irow = 0;

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i === j) { b[i * n + j] = 1; } else { b[i * n + j] = 0; }
      }
    }
    for (let j = 0; j < n; j++) { ipiv[j] = 0; }
    for (let i = 0; i < n; i++) {
      let big = 0;
      for (let j = 0; j < n; j++) {
        if (ipiv[j] !== 1) {
          for (let k = 0; k < n; k++) {
            if (ipiv[k] === 0) {
              if (Math.abs(X[j * n + k]) >= big) {
                big = Math.abs(X[j * n + k]);
                irow = j;
                icol = k;
              }
            }
          }
        }
      }
      ++(ipiv[icol]);

      if (irow !== icol) {
        for (let l = 0; l < n; l++) {
          const temp = X[irow * n + l];
          X[irow * n + l] = X[icol * n + l];
          X[icol * n + l] = temp;
        }
        for (let l = 0; l < m; l++) {
          const temp = b[irow * n + l];
          b[irow * n + l] = b[icol * n + l];
          b[icol * n + l] = temp;
        }
      }
      indxr[i] = irow;
      indxc[i] = icol;

      if (X[icol * n + icol] === 0) { return false; } // Singular

      const pivinv = 1 / X[icol * n + icol];
      X[icol * n + icol] = 1;
      for (let l = 0; l < n; l++) { X[icol * n + l] *= pivinv; }
      for (let l = 0; l < m; l++) { b[icol * n + l] *= pivinv; }

      for (let ll = 0; ll < n; ll++) {
        if (ll !== icol) {
          const dum = X[ll * n + icol];
          X[ll * n + icol] = 0;
          for (let l = 0; l < n; l++) { X[ll * n + l] -= X[icol * n + l] * dum; }
          for (let l = 0; l < m; l++) { b[ll * n + l] -= b[icol * n + l] * dum; }
        }
      }
    }
    for (let l = (n - 1); l >= 0; l--) {
      if (indxr[l] !== indxc[l]) {
        for (let k = 0; k < n; k++) {
          const temp = X[k * n + indxr[l]];
          X[k * n + indxr[l]] = X[k * n + indxc[l]];
          X[k * n + indxc[l]] = temp;
        }
      }
    }

    return true;
  }
  // Variogram models
  private kriging_variogram_gaussian(h, nugget, range, sill, A) {
    return nugget + ((sill - nugget) / range) *
      (1.0 - Math.exp(-(1.0 / A) * Math.pow(h / range, 2)));
  }
  private kriging_variogram_exponential(h, nugget, range, sill, A) {
    return nugget + ((sill - nugget) / range) *
      (1.0 - Math.exp(-(1.0 / A) * (h / range)));
  }
  private kriging_variogram_spherical(h, nugget, range, sill, A) {
    if (h > range) { return nugget + (sill - nugget) / range; }
    return nugget + ((sill - nugget) / range) *
      (1.5 * (h / range) - 0.5 * Math.pow(h / range, 3));
  }
}

declare global {
  interface Array<T> {
    rep(n: number): number[];
    pip(x: number, y: number): boolean;
    mean(): number;
    max(): number;
    min(): number;
  }
}

// 扩展array方法 不能实用箭头函数
Array.prototype.max = function() {
  return Math.max.apply(null, this);
};
Array.prototype.min = function() {
  return Math.min.apply(null, this);
};
Array.prototype.mean = function() {
  let sum = 0;
  // tslint:disable-next-line: prefer-for-of
  for (let i = 0; i < this.length; i++) {
    sum += this[i];
  }
  return sum / this.length;
};
Array.prototype.rep = function(n) {
  return Array.apply(null, new Array(n))
    .map(Number.prototype.valueOf, this[0]);
};
Array.prototype.pip = function(x, y) {
  let i = 0;
  let j = 0;
  let c = false;
  for (i = 0, j = this.length - 1; i < this.length; j = i++) {
    if (((this[i][1] > y) !== (this[j][1] > y)) &&
      (x < (this[j][0] - this[i][0]) * (y - this[i][1]) / (this[j][1] - this[i][1]) + this[i][0])) {
      c = !c;
    }
  }
  return c;
};
