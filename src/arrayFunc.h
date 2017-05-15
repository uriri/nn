/*
 * arrayFunc.h
 *
 *  Created on: 2017/02/14
 *      Author: haga
 */

#ifndef ARRAYFUNC_H_
#define ARRAYFUNC_H_

#include <array>
#include <utility>
#include <vector>
#include <iterator>

//生配列をarrayに変換
//index_tupleイディオムとかいうやつを使ってる
template< std::size_t... indices >
struct index_tuple{};

template < std::size_t frist,
			std::size_t last,
			class result = index_tuple<>,
			bool finish = (frist>=last) >
struct index_range{
    using type = result;
};

template < std::size_t frist,
			std::size_t last,
			std::size_t... indices >
struct index_range< frist, last, index_tuple<indices...>, false >
         : index_range<frist+1, last, index_tuple<indices..., frist>>
{};

template <class T, std::size_t N, std::size_t... indices>
constexpr std::array<T, N> toArrayImpl( T const(& native_array)[N], index_tuple<indices...>){
    return {{ native_array[indices]... }};
}

template <class T, std::size_t N>
constexpr std::array<T, N> toArray( T const(& native_array)[N]){
    return toArrayImpl( native_array, typename index_range<0, N>::type() );
}

//vectorをn分割する
void divide(std::vector< std::vector<int> >& data, const std::vector<int>& res, unsigned int n){
	const unsigned int len = res.size();
	const unsigned int partialLen = (len + n - 1) / n;
	data.resize(partialLen);
	unsigned int resIndex = 0;
	for(unsigned int i=0; i<partialLen; ++i){
		data[i].reserve(n);
		for(unsigned int j=0; j<n; ++j, ++resIndex){
			if(resIndex == len){
				data[i].shrink_to_fit();
				return;
			}
			data[i].emplace_back(res[resIndex]);
		}
	}
}

//整数列 /* 0, 1, 2 ... s */を作る
//vector.size()はnon-constexprなので使用不可
template<typename T, size_t N>
struct range{
	T value[N];
	constexpr range():value(){
		for(unsigned int i=0; i<N; ++i)
			value[i] = i;
	}
};

#endif /* ARRAYFUNC_H_ */
