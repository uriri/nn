/*
 * NeuralNetwork.h
 *
 *  Created on: 2015/12/03
 *      Author: haga
 */

#ifndef BONANZA_NEURALNETWORK_H_
#define BONANZA_NEURALNETWORK_H_

#include <vector>
#include <random>
#include <fstream>

namespace NN {

constexpr double alpha = 1.0; //シグモイド関数のゲイン

//入力数, 中間層
//出力は1つに固定
class NeuralNetwork {
private:
	std::vector<double> m_input;
	std::vector<double> m_hiddenLayer;					//中間層の内部,sigmoidは通してない
	double m_output;									//出力層の内部,sigmoidは通してない

	std::vector< std::vector<double> > m_weightVecI2H;	//重みベクトル（入力->中間）[hide][input]
	std::vector<double> m_weightVecH2O;					//重みベクトル（中間->出力）
	unsigned int m_weightSize;							//重みの数
	std::random_device m_rndDevice;

	double sigmoid(double arg) {
		return (1.0 / (1.0 + exp(-alpha * arg)));
	}

	double d_sigmoid(double arg) {
		return ( (1.0 - sigmoid(arg)) * sigmoid(arg) );
	}

	void reset() {
		//中間層の内部リセット
		for (auto& hide : m_hiddenLayer) {
			hide = 0.0;
		}
		m_output = 0.0;
	}

public:
	NeuralNetwork(int inSize, int hideSize) {
		//バイアスを含んだ入力，中間層の数
		const int iqty = inSize + 1;
		const int hqty = hideSize + 1;

		m_weightSize = iqty*hideSize + hqty;

		m_input.resize(iqty);
		m_hiddenLayer.resize(hqty);

		m_weightVecI2H.resize(hideSize);
		for (int i = 0; i < hideSize; ++i) {
			m_weightVecI2H[i].resize(iqty);
		}

		m_weightVecH2O.resize(hqty);

		//biasの設定
		m_input[inSize] = -1;
		m_hiddenLayer[hideSize] = -1;
		init();
	}

	virtual ~NeuralNetwork() { }

	void init() {
		std::mt19937 mt(m_rndDevice());
		std::uniform_real_distribution<> randWeight(-1.0, 1.0);

		int t = 0;
		//入力->中間
		for (int i = 0; i < m_weightVecI2H.size(); ++i) {
			for (int j = 0; j < m_weightVecI2H[i].size(); ++j) {
				m_weightVecI2H[i][j] = randWeight(mt);
			}
		}
		//中間->出力
		for (int i = 0; i < m_weightVecH2O.size(); ++i) {
			m_weightVecH2O[i] = randWeight(mt);
		}
		reset();
	}

	void readWeightFile(const std::string& file, const int size) {
		std::ifstream ifs;
		ifs.open(file.c_str(), std::ios::binary);
		std::vector<double> ww(size);
		if (!ifs.fail()) {
			int readSize;
			ifs.read(reinterpret_cast<char*>(&readSize), sizeof(readSize));
			ifs.read(reinterpret_cast<char*>(&ww[0]), readSize * sizeof(double));
		}
		setWeight(ww);
		ifs.close();
	}

	void setWeight(const std::vector<double>& arg){
		const int& i2hSize = m_weightVecI2H.size();
		int h2oPos = m_weightVecI2H.size()*m_weightVecI2H[0].size();
		for(int i=0; i<i2hSize; ++i){
			for(int j=0; j<m_weightVecI2H[i].size(); ++j){
				m_weightVecI2H[i][j] = arg[i*i2hSize+j];
			}
		}
		for(int i=0; i<m_weightVecH2O.size(); ++i){
			m_weightVecH2O[i] = arg[i + h2oPos];
		}
	}

	//入力をもらって出力を計算
	template<int Size>
	void calOutPut(int (&input)[Size]) {
		for (int i = 0; i < Size; ++i) {
			m_input[i] = static_cast<double>(input[i]);
		}
		reset();
		//入力->中間
		for (int i = 0; i < m_weightVecI2H.size(); ++i) {
			for (int j = 0; j < m_weightVecI2H[i].size(); ++j) {
				m_hiddenLayer[i] += (m_weightVecI2H[i][j] * m_input[j]);
			}
		}
		//中間->出力
		for (int i = 0; i < m_weightVecH2O.size(); ++i) {
			m_output += (m_weightVecH2O[i] * sigmoid(m_hiddenLayer[i]));
		}
	}

	//出力を返す
	double getOutPut(){
		return sigmoid(m_output);
	}

	//重みを1次元ベクトルに変換
	std::vector<double> getWeightOneDim() {
		std::vector<double> tmp(m_weightSize);
		int h2oPos = m_weightVecI2H.size()*m_weightVecI2H[0].size();

		for(int i=0; i<m_weightVecI2H.size(); ++i){
			for(int j=0; j<m_weightVecI2H[0].size(); ++j){
				 tmp[i*m_weightVecI2H[0].size()+j] = m_weightVecI2H[i][j];
			}
		}
		for(int i=0; i<m_weightVecH2O.size(); ++i){
			tmp[i + h2oPos] = m_weightVecH2O[i];
		}
		return tmp;
	}

	//勾配の計算
	std::vector<double> getGrad(){
		std::vector<double> gradVec(m_weightSize);
		int h2oPos = m_weightVecI2H.size()*m_weightVecI2H[0].size();

		//入力->中間
		for (int i = 0; i < m_weightVecI2H.size(); ++i) {
			for (int j = 0; j < m_weightVecI2H[i].size(); ++j) {
				gradVec[i*m_weightVecI2H[0].size()+j] = d_sigmoid(m_output)*m_weightVecH2O[i+h2oPos]*d_sigmoid(m_hiddenLayer[i])*m_input[j];
			}
		}
		//中間->出力
		for (int i = 0; i < m_weightVecH2O.size(); ++i) {
			gradVec[i + h2oPos] = d_sigmoid(m_output)*sigmoid(m_hiddenLayer[i]);
		}
		return gradVec;
	}

};

} /* namespace NN */

#endif /* BONANZA_NEURALNETWORK_H_ */
