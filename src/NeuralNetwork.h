/*
 * NeuralNetwork.h
 *
 *  Created on: 2015/12/03
 *      Author: haga
 */

#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include <vector>
#include <random>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <memory>

#include <eigen3/Eigen/Core>

namespace Eigen {

//バイナリでEinge::Matrixを出力
template<class Matrix>
void writeBinary(const std::string& fileName, const Matrix& mat){
	std::ofstream ofs(fileName, std::ios::binary|std::ios::trunc);
	typename Matrix::Index rows=mat.rows(), cols=mat.cols();
	ofs.write((char*)(&rows), sizeof(typename Matrix::Index));
	ofs.write((char*)(&cols), sizeof(typename Matrix::Index));
	ofs.write((char*)mat.data(), rows*cols*sizeof(typename Matrix::Scalar));
	ofs.close();
}

//バイナリでEinge::Matrixを入力
template<class Matrix>
void readBinary(const std::string& fileName, Matrix& mat){
	std::ifstream ifs(fileName, std::ios::binary);
	typename Matrix::Index rows=0, cols=0;
	ifs.read((char*)(&rows), sizeof(typename Matrix::Index));
	ifs.read((char*)(&cols), sizeof(typename Matrix::Index));
	mat.resize(rows, cols);
	ifs.read((char*)(mat.data()), rows*cols*sizeof(typename Matrix::Scalar));
	ifs.close();
}

} //Eigen:;

namespace NN {

//活性化関数インターフェース
class ActiveFunc{
public:
	virtual ~ActiveFunc() = default;
	virtual double val(const double arg) = 0;		//活性化関数の値
	virtual double d_val(const double arg) = 0;		//導関数の値
};

//シグモイド
class Sigmoid : public ActiveFunc {
private:
	static constexpr double alpha = 1.0; //シグモイド関数のゲイン
public:
	double val(const double arg) override {
		return (1.0 / (1.0 + std::exp(-alpha * arg)));
	}
	double d_val(const double arg) override {
		return ( (1.0 - val(arg)) * val(arg) );
	}
};

//ランプ関数
class ReLU : public ActiveFunc {
public:
	double val(const double arg) override {
		return std::max(arg, 0.0);
	}
	double d_val(const double arg) override {
		return arg<0.0 ? 0.0 : 1.0;
	}
};

//Eigenを使ったNN
//入力，中間にバイアスはなしで
template<std::size_t inSize, std::size_t hideSize, std::size_t outSize>
class NeuralNetwork {
private:
	unsigned int m_weightSize;		//重みの数

	Eigen::VectorXd m_input;		//入力層
	Eigen::VectorXd m_hiddenLayer;	//中間層の出力,sigmoidは通していない
	Eigen::VectorXd m_output;		//出力層の出力,sigmoidは通していない

	Eigen::MatrixXd m_weightI2H;	//重み行列（入力->中間）
	Eigen::MatrixXd m_weightH2O;	//重み行列（中間->出力）

	std::unique_ptr<ActiveFunc> m_hideActFunc;			//中間層の活性化関数
	std::unique_ptr<ActiveFunc> m_outActFunc;			//出力層の活性化関数

	//ニューラルネット出力計算
	constexpr void calOutPut_imp(){
		//入力->中間
		m_hiddenLayer.head(hideSize) = m_input.transpose()*m_weightI2H;

		//中間->出力
		m_output = m_hiddenLayer.unaryExpr(
				[this](double x){
					return m_hideActFunc->val(x);
				}
		).transpose()*m_weightH2O;
	};

public:
	constexpr NeuralNetwork():
		m_weightSize( ((inSize+1)*hideSize)+((hideSize+1)*outSize) ),
		m_input(Eigen::VectorXd::Zero(inSize+1)),
		m_hiddenLayer(Eigen::VectorXd::Zero(hideSize+1)),
		m_output(Eigen::VectorXd::Zero(outSize)),
		m_weightI2H(Eigen::MatrixXd::Random(inSize+1, hideSize)),
		m_weightH2O(Eigen::MatrixXd::Random(hideSize+1, outSize)),
		m_hideActFunc(std::make_unique<ReLU>()),
		m_outActFunc(std::make_unique<Sigmoid>())
	{
		m_input(0) = -1.0;
		m_hiddenLayer(0) = -1.0;
	}

	constexpr NeuralNetwork(const std::string& fileI2H, const std::string& fileH2O):
		m_weightSize( ((inSize+1)*hideSize)+((hideSize+1)*outSize) ),
		m_input(Eigen::VectorXd::Zero(inSize+1)),
		m_hiddenLayer(Eigen::VectorXd::Zero(hideSize+1)),
		m_output(Eigen::VectorXd::Zero(outSize)),
		m_weightI2H(Eigen::MatrixXd::Random(inSize+1, hideSize)),
		m_weightH2O(Eigen::MatrixXd::Random(hideSize+1, outSize)),
		m_hideActFunc(std::make_unique<ReLU>()),
		m_outActFunc(std::make_unique<Sigmoid>())
	{
		m_input(0) = -1.0;
		m_hiddenLayer(0) = -1.0;
		Eigen::readBinary(fileI2H, m_weightI2H);
		Eigen::readBinary(fileH2O, m_weightH2O);
	}

	virtual ~NeuralNetwork() = default;

	//入力をもらって出力を計算
	template<class T>
	void setInput(const T& input){
		std::array<double, inSize> dIn;
		for(int i=0; i<inSize; ++i){
			dIn[i] = static_cast<double>(input[i]);
		}
		m_input.tail(inSize) = Eigen::Map<Eigen::VectorXd>(&dIn[0], inSize);
		calOutPut_imp();
	}

	void setInput(const Eigen::VectorXd& input) {
		m_input.tail(inSize) = input;
		calOutPut_imp();
	}

	//重みを1次元ベクトルに変換
	void getWeightOneDim(Eigen::VectorXd& arg) {
		arg.resize(m_weightSize);
		arg.head(m_weightI2H.size()) = Eigen::Map<Eigen::VectorXd>(m_weightI2H.data(), m_weightI2H.size());
		arg.tail(m_weightH2O.size()) = Eigen::Map<Eigen::VectorXd>(m_weightH2O.data(), m_weightH2O.size());
	}

	//重みセット
	void setWeight(Eigen::VectorXd& arg){
		m_weightI2H = Eigen::Map<Eigen::MatrixXd>(arg.head(m_weightI2H.size()).data(),
				m_weightI2H.rows(), m_weightI2H.cols());
		m_weightH2O = Eigen::Map<Eigen::MatrixXd>(arg.tail(m_weightH2O.size()).data(),
				m_weightH2O.rows(), m_weightH2O.cols());
	}

	//出力
	double getOutPut(int index = 0) const { return m_outActFunc->val(m_output(index)); }
	Eigen::VectorXd getOutVec() const {
		auto f = [this](double x){ return m_outActFunc->val(x); };
		return m_output.unaryExpr(f);
	}

	//重みgetter
	Eigen::MatrixXd getWeightI2H_Mat() const { return m_weightI2H; }
	Eigen::MatrixXd getWeightH2O_Mat() const { return m_weightH2O; }

	//勾配の計算
	void getGrad(Eigen::VectorXd& grad) const {
		grad.resize(m_weightSize);
		//入力->中間
		grad.head(m_weightI2H.size()) = getGradI2H();
		//中間->出力
		grad.tail(m_weightH2O.size()) = getGradH2O();
	}

	//勾配の計算
	//ほんとは行列計算的に求めたいけどとりあえず配列の時のやつそのままで
	Eigen::VectorXd getGradI2H() const {
		const int is = m_weightI2H.rows();
		const int hs = m_weightI2H.cols();
		const int os = m_output.size();
		Eigen::VectorXd tmp = Eigen::VectorXd::Zero(m_weightI2H.size());
		for (int i=0; i<hs; ++i) {
			for (int j=0; j<is; ++j) {
				tmp(i*is+j) = m_hideActFunc->d_val(m_hiddenLayer[i])*m_input[j];
				for(int k=0; k<os; ++k){
					tmp(i*is+j) *= m_outActFunc->d_val(m_output(k))*m_weightH2O(i, k);
				}
			}
		}
		return tmp;
	}

	//勾配の計算
	Eigen::VectorXd getGradH2O() const {
		auto o = [this](double x){ return m_outActFunc->d_val(x); };
		auto h = [this](double x){ return m_hideActFunc->val(x); };
		Eigen::MatrixXd tmp = m_output.unaryExpr(o)*(m_hiddenLayer.unaryExpr(h)).transpose();
		return Eigen::Map<Eigen::VectorXd>(tmp.data(), m_weightH2O.size());
	}

	//誤差逆伝播法による重みの修正
	void backPropagation(const Eigen::VectorXd& teach, const double rate) {
		Eigen::VectorXd delta_k = m_output - teach;
		Eigen::VectorXd delta_j = m_weightH2O.transpose() * delta_k;
		auto f = [this](double x){ return m_hideActFunc->val(x); };

		delta_j = delta_j.array() * m_hiddenLayer.unaryExpr(f).array();

		m_weightH2O.array() -= rate * (delta_k * m_output.transpose()).array();
		m_weightI2H.array() -= rate * (delta_j.tail(m_hiddenLayer.size()) * m_input.transpose()).array();
	}

	/*************
	 * いらない群
	 *************/
//	void setWeightI2H(Eigen::VectorXd arg){
//		m_weightI2H = Eigen::Map<Eigen::MatrixXd>(arg.data(), m_weightI2H.cols(), m_weightI2H.rows());
//	}
//
//	void setWeightH2O(Eigen::VectorXd arg){
//		m_weightH2O = Eigen::Map<Eigen::MatrixXd>(arg.data(), m_weightH2O.cols(), m_weightH2O.rows());
//	}
//
//	Eigen::VectorXd getWeightI2H() const {
//		return Eigen::Map<Eigen::VectorXd>(m_weightI2H.data(), m_weightI2H.cols()*m_weightI2H.rows());
//	}
//
//	Eigen::VectorXd getWeightH2O() const {
//		return Eigen::Map<Eigen::VectorXd>(m_weightH2O.data(), m_weightH2O.cols()*m_weightH2O.rows());
//	}
};

}/* namespace NN */

#endif /* NEURALNETWORK_H_ */
