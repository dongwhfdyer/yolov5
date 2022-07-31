let mapleader=" "
":map lv mJ
set relativenumber
set nu
:nnoremap lj  ^v$
:nnoremap lk  $v0
:nnoremap dc  vF#d
:nnoremap lz %
:nnoremap lx f(l
:vnoremap lf "+ye
:nnoremap lp "+p
:vnoremap lp "+p
:nnoremap lc "+yawe
:vnoremap la s()<ESC>hp%
:nnoremap lq viw"+p
:nnoremap ll diw

set smartcase                   " no ignorecase if Uppercase char present
set incsearch                   " search is incremental
set hlsearch                    " highlight search

"nnoremap <Leader>y ma"+yiw"ayiw
nnoremap <Leader>y "+yy"ayy
nnoremap <Leader>a ci"
nnoremap <Leader>d di(
" Window operation

" (关于窗口操作)
"分屏光标跳上面；
nnoremap <Leader>ww <C-W>
"清空窗口
nnoremap <Leader>wd <C-W>c
nnoremap <Leader>q <C-W>c
"下面屏幕光标跳上面
nnoremap sh <C-W>h
nnoremap sl <C-W>l
nnoremap sj <C-W>j
nnoremap sk <C-W>k
"上下分屏
nnoremap ss <C-W>s
"在加个一个分屏
nnoremap <Leader>w- <C-W>s
"横向分屏
nnoremap sv <C-W>v "
nnoremap sv <C-W>v
nnoremap go gT
nnoremap gp gt
"nnoremap <Leader><Leader> $a#
"跳转到实体类
"save
nnoremap <Leader>q :q<CR>
"跳转到实现类
"right click
nnoremap do d$
nnoremap du dt(
nnoremap dm dt)
nnoremap dy dT(
nnoremap dp diW


nnoremap vo v$
nnoremap vu vt(
nnoremap vm vt)
nnoremap vy vT(
nnoremap vp viW
nnoremap vQ aprint("########################################")<ESC>
nnoremap vq a##########nhuk####################################<ESC>
vnoremap vz d10a#<ESC>anhuk<ESC>36a#<ESC>o<ESC>po<ESC>10a#<ESC>anhuk<ESC>36a#<ESC>



" add try catch finally
vnoremap lg datry:<CR><ESC>poexcept Exception as e:<CR>print(e)<ESC>

" if els
vnoremap lr daif True:<CR><ESC>p
