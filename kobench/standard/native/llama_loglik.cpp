// llama_loglik — GGUF 모델로 (문맥, 연속) 토큰열의 로그우도를 계산한다 (lm-eval loglikelihood 요청용).
//
// 사용: llama_loglik -m model.gguf -i requests.jsonl -o results.jsonl [--n-ctx 65536] [--n-batch 4096] [--ngl 999]
// 입력 한 줄: {"ctx":[토큰...], "conts":[[토큰...], ...]}   (ctx는 비어 있으면 안 된다)
// 출력 한 줄(같은 순서): {"ll":[연속별 로그우도 합], "greedy":[연속별 greedy 일치 여부]}
//
// 같은 문맥을 공유하는 연속들은 문맥 KV를 한 번만 계산하고 seq_cp로 복사한다(kv_unified).
// 여러 요청을 한 배치에 묶는다(시퀀스 id 최대 255, 토큰 예산 n_ctx).
// 로그우도는 각 위치 로짓의 log_softmax에서 목표 토큰 값(float32, 로짓은 llama.cpp 출력 그대로).
#include "llama.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using json = nlohmann::json;

struct Req {
    std::vector<llama_token> ctx;
    std::vector<std::vector<llama_token>> conts;
    std::vector<double> ll;
    std::vector<int> greedy;
};

static void score_logits(const float * l, int n_vocab, llama_token tgt, double & ll, int & greedy) {
    float mx = l[0];
    int am = 0;
    for (int v = 1; v < n_vocab; v++) {
        if (l[v] > mx) { mx = l[v]; am = v; }
    }
    double s = 0.0;
    for (int v = 0; v < n_vocab; v++) s += std::exp((double) l[v] - mx);
    ll += (double) l[tgt] - mx - std::log(s);
    if (am != tgt) greedy = 0;
}

int main(int argc, char ** argv) {
    std::string model_path, in_path, out_path;
    int n_ctx = 65536, n_batch = 4096, ngl = 999;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "-m") model_path = next();
        else if (a == "-i") in_path = next();
        else if (a == "-o") out_path = next();
        else if (a == "--n-ctx") n_ctx = std::stoi(next());
        else if (a == "--n-batch") n_batch = std::stoi(next());
        else if (a == "--ngl") ngl = std::stoi(next());
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    if (model_path.empty() || in_path.empty() || out_path.empty()) {
        fprintf(stderr, "usage: llama_loglik -m model.gguf -i in.jsonl -o out.jsonl\n");
        return 2;
    }
    std::vector<Req> reqs;
    {
        std::ifstream f(in_path);
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            json j = json::parse(line);
            Req r;
            r.ctx = j["ctx"].get<std::vector<llama_token>>();
            r.conts = j["conts"].get<std::vector<std::vector<llama_token>>>();
            if (r.ctx.empty()) { fprintf(stderr, "empty ctx\n"); return 3; }
            r.ll.assign(r.conts.size(), 0.0);
            r.greedy.assign(r.conts.size(), 1);
            reqs.push_back(std::move(r));
        }
    }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 4; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = n_batch;
    cp.n_ubatch = std::min(n_batch, 2048);
    cp.n_seq_max = 256;
    cp.kv_unified = true;
    cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "context init failed\n"); return 5; }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_memory_t mem = llama_get_memory(ctx);
    llama_batch batch = llama_batch_init(n_batch, 0, 1);

    // 배치에 넣을 토큰: (토큰, 위치, seq, logits 필요, 콜백 대상)
    struct Tok { llama_token t; llama_pos p; llama_seq_id s; bool want; int req; int cont; int k; };
    auto run_tokens = [&](const std::vector<Tok> & toks) -> bool {
        // n_batch씩 나눠 decode하고, 로짓이 필요한 위치마다 점수를 낸다
        for (size_t o = 0; o < toks.size(); o += n_batch) {
            size_t e = std::min(toks.size(), o + (size_t) n_batch);
            batch.n_tokens = 0;
            for (size_t i = o; i < e; i++) {
                const Tok & x = toks[i];
                int b = batch.n_tokens++;
                batch.token[b] = x.t; batch.pos[b] = x.p; batch.n_seq_id[b] = 1; batch.seq_id[b][0] = x.s; batch.logits[b] = x.want;
            }
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return false; }
            for (size_t i = o; i < e; i++) {
                const Tok & x = toks[i];
                if (!x.want) continue;
                const float * l = llama_get_logits_ith(ctx, (int32_t) (i - o));
                Req & r = reqs[x.req];
                if (x.cont < 0) {             // 문맥 마지막 토큰: 모든 연속의 첫 토큰 점수
                    for (size_t c = 0; c < r.conts.size(); c++) score_logits(l, n_vocab, r.conts[c][0], r.ll[c], r.greedy[c]);
                } else {
                    score_logits(l, n_vocab, r.conts[x.cont][x.k + 1], r.ll[x.cont], r.greedy[x.cont]);
                }
            }
        }
        return true;
    };

    size_t i = 0;
    const int budget = n_ctx - 16;
    while (i < reqs.size()) {
        // 묶음 구성: 시퀀스 id와 KV 셀 예산 안에서
        size_t j = i;
        int seqs = 0, cells = 0;
        while (j < reqs.size()) {
            int need_s = 1 + (int) reqs[j].conts.size();
            int need_c = (int) reqs[j].ctx.size();
            for (auto & c : reqs[j].conts) need_c += (int) c.size();
            if (j > i && (seqs + need_s > 255 || cells + need_c > budget)) break;
            seqs += need_s; cells += need_c; j++;
        }
        if (cells > budget) { fprintf(stderr, "request %zu too long (%d cells)\n", i, cells); return 6; }
        llama_memory_clear(mem, true);
        std::vector<Tok> ph1, ph2;
        int sid = 0;
        std::vector<int> base(j - i);
        for (size_t r = i; r < j; r++) {
            base[r - i] = sid;
            const auto & c = reqs[r].ctx;
            for (size_t k = 0; k < c.size(); k++)
                ph1.push_back({c[k], (llama_pos) k, sid, k + 1 == c.size(), (int) r, -1, 0});
            sid += 1 + (int) reqs[r].conts.size();
        }
        if (!run_tokens(ph1)) return 7;
        for (size_t r = i; r < j; r++) {
            const int s0 = base[r - i];
            const llama_pos n = (llama_pos) reqs[r].ctx.size();
            for (size_t c = 0; c < reqs[r].conts.size(); c++) {
                const auto & ct = reqs[r].conts[c];
                if (ct.size() < 2) continue;
                const llama_seq_id sc = s0 + 1 + (int) c;
                llama_memory_seq_cp(mem, s0, sc, -1, -1);
                for (size_t k = 0; k + 1 < ct.size(); k++)
                    ph2.push_back({ct[k], n + (llama_pos) k, sc, true, (int) r, (int) c, (int) k});
            }
        }
        if (!ph2.empty() && !run_tokens(ph2)) return 8;
        i = j;
        fprintf(stderr, "\r[llama_loglik] %zu/%zu", i, reqs.size());
    }
    fprintf(stderr, "\n");
    std::ofstream out(out_path);
    for (auto & r : reqs) {
        json j;
        j["ll"] = r.ll;
        j["greedy"] = r.greedy;
        out << j.dump() << "\n";
    }
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
