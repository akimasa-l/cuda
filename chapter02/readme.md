# chapter 2を読みながらの感想

- `alloc`とか`free`とか、メモリを確保するのにそんな関数使ってるのおそすぎませんか！？時代はオブジェクト指向ですよ！？低レベル言語嫌いです！！(過激思想)
- gridとかblockとか、2次元とか3次元とかって、概念多すぎてわけわかんないですよ！
- `__global__`って必ずGPUから呼び出さなきゃいけないんですね.
- GPUで呼び出す関数は(GPUの中では)書かれた順番通りに動作するって聞いて、安心しました。
- なんだか2つ全く異なるコードを一つの関数の中で書いていて、頭がこんがらがりそうになります。
- `nvprof`は`./a.out`みたいに`./`をつけなきゃいけないの、初見殺し
- `nvprof`呼び出したけどあんまりいい感じにできてない？`sudo`が必要？
- メモリ確保にすっごい時間がかかる
- なんか2080と1080両方使えるって言ってたけど1個しかDetectできてないな[checkDeviceInfor.log](checkDeviceInfor.log)
- もしかしてポート番号を変えないとGPU変えられない？よくわからない
- ポート番号変えたらGTX1080使えたわ　まあ俺はRTX2080を使い続けるけどな！w
- 75ページの練習問題1個目に書いてあるとおりに1024から1023に変えたんですけどなんか変わりました？何も変わってないと思ってるんですが
- ちょっと実行時間が短くなったぐらいですか？ [該当するcommit](https://github.com/akimasa-l/cuda/commit/39c78da0950876a916bf828e71180cd9f3c028af?branch=39c78da0950876a916bf828e71180cd9f3c028af&diff=split)
- 練習問題2個目は問題文がよくわからなかたのでSolutionを見てみる　実行時間がめっちゃ伸びた

## logの残し方

`nvprof ./***.out |& tee ./***.log`で結果を見ながらlogに残せる

だけどこれが何故かstdoutが表示されないしlogにものこせてない謎の状態が出る
