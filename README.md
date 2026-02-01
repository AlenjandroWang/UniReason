<p align="center">
  <img src="unireason_lo.png" alt="BAGEL" width="450"/ hei>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2505.14683">
    <img
      src="https://img.shields.io/badge/UniReason-Paper-red?logo=arxiv&logoColor=red"
      alt="UniReason Paper on arXiv"
    />
  </a>
  <a href="https://huggingface.co/Alex11556666/UniReason">
    <img 
        src="https://img.shields.io/badge/UniReason-Model-yellow?logo=huggingface&logoColor=yellow" 
        alt="UniReason Model"
    />
  </a>
  <a href="https://huggingface.co/spaces/ByteDance-Seed/BAGEL">
    <img 
        src="https://img.shields.io/badge/UniReason-Data-yellow?logo=huggingface&logoColor=yellow" 
        alt="UniReason tuing data"
    />
  </a>
</p>

# UniReason 1.0: A Unified Reasoning Framework for World Knowledge Aligned Image Generation and Editing
> [Dianyi Wang**](https://scholar.google.com/citations?hl=zh-CN&user=iP2HPFEAAAAJ), [Chaofan Ma**](https://github.com/chaofanma), Feng Han, [Size Wu](https://wusize.github.io/), [Wei Song](https://scholar.google.com/citations?hl=zh-CN&user=k0blIUIAAAAJ), [Yibin Wang](https://codegoat24.github.io/), [Zhixiong Zhang](https://rookiexiong7.github.io/), Tianhang Wang, [Siyuan Wang :email:](https://siyuanwangw.github.io/), [Zhongyu Wei :email:](http://www.fudan-disc.com/people/zywei), [Jiaqi Wang :tophat: :email: ](https://myownskyw7.github.io/)
>> contact: dywang24@m.fudan.edu.cn, sw_641@usc.edu, Wei-zywei@fudan.edu.cn, wjqdev@gmail.com

>> We propose **UniReason**, a unified framework
that harmonizes these two tasks through a dual reasoning paradigm. We formulate generation
as world knowledge-enhanced planning to inject implicit constraints, and leverage editing capabilities
for fine-grained visual refinement to further correct visual errors via self-reflection. This approach
unifies generation and editing within a shared representation, mirroring the human cognitive process
of planning followed by refinement. We support this framework by systematically constructing a
large-scale reasoning-centric dataset covering five major knowledge domains (e.g.,
cultural commonsense, physics, etc.) for planning, alongside an agent-generated corpus for visual
self-correction. Extensive experiments demonstrate that UniReason achieves advanced performance
on reasoning-intensive benchmarks such as WISE and KrisBench, while maintaining superior general
synthesis capabilities on GenEval and ImgEdit. The figure below showcases UniReason's qualitative performance.

<p align="center"><img src="unireason.png" width="95%"></p>
