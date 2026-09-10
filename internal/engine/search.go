package engine

import (
	"context"
	"log/slog"
	"strings"

	"github.com/anatolykoptev/go-engine/search"
	"github.com/anatolykoptev/go-kit/env"
	"golang.org/x/time/rate"
)

// FilterByScore removes results below minScore, keeping at least minKeep.
func FilterByScore(results []SearxngResult, minScore float64, minKeep int) []SearxngResult {
	return search.FilterByScore(results, minScore, minKeep)
}

// DedupByDomain limits results to maxPerDomain per domain.
func DedupByDomain(results []SearxngResult, maxPerDomain int) []SearxngResult {
	return search.DedupByDomain(results, maxPerDomain)
}

// RawSearcher is the interface for general-purpose web search via go-search.
// When wired (SetRawSearcher), SearchWeb tries go-search first (which has
// Brave API + ox-browser + DDG through proxy), falling back to SearchDirect
// (local direct scrapers) on error or when nil.
type RawSearcher interface {
	RawSearch(ctx context.Context, query string) ([]SearxngResult, error)
}

// rawSearcherInstance is the optional go-search-backed web search backend.
// When non-nil, SearchWeb uses it as the primary path; SearchDirect is the
// fallback. Wired from main.go when GO_SEARCH_URL is set.
var rawSearcherInstance RawSearcher

// SetRawSearcher wires the go-search raw search client at startup.
func SetRawSearcher(r RawSearcher) { rawSearcherInstance = r }

// SearchWeb tries go-search first (if wired), falling back to SearchDirect.
// This routes search through go-search's fused multi-source pipeline
// (Brave API + ox-browser + DDG via proxy) instead of hitting DDG
// directly from the container, which gets 202-blocked from a datacenter IP.
func SearchWeb(ctx context.Context, query, language string) []SearxngResult {
	if rawSearcherInstance != nil {
		results, err := rawSearcherInstance.RawSearch(ctx, query)
		if err == nil {
			return results
		}
		slog.Debug("SearchWeb: go-search error, falling back to SearchDirect",
			slog.Any("error", err))
	}
	return SearchDirect(ctx, query, language)
}

// SearchDirect queries enabled direct scrapers in parallel.
// Returns merged results from all direct sources. Failures are non-fatal.
// Returns nil when the engine is not initialized (fetcherProxy == nil).
//
// Discards per-leg DirectStats — use SearchDirectWithStats if you need
// the degraded-mode signal (Attempted > 0 && OK == 0).
func SearchDirect(ctx context.Context, query, language string) []SearxngResult {
	results, _ := SearchDirectWithStats(ctx, query, language)
	return results
}

// SearchDirectWithStats is like SearchDirect but also returns DirectStats
// from the upstream fan-out. The primary signal: Attempted > 0 && OK == 0
// means every launched leg was blocked or failed — the DC-IP / censorship
// degraded mode that is otherwise indistinguishable from genuine zero results.
func SearchDirectWithStats(ctx context.Context, query, language string) ([]SearxngResult, search.DirectStats) {
	if fetcherProxy == nil {
		return nil, search.DirectStats{}
	}
	results, stats := search.SearchDirect(ctx, directSearchConfig(), query, language)
	return results, stats
}

// directBrowser returns the best available BrowserDoer for direct scrapers.
// Prefers DirectClient (no-proxy Chrome-TLS, built when FETCH_DIRECT_FIRST is set)
// and falls back to BrowserClient (proxy-backed). It performs concrete-pointer
// nil checks before converting to the BrowserDoer interface so a typed-nil
// *BrowserClient never escapes as a non-nil interface value.
func directBrowser() search.BrowserDoer {
	if fetcherProxy == nil {
		return nil
	}
	if dc := fetcherProxy.DirectClient(); dc != nil {
		return dc
	}
	if bc := fetcherProxy.BrowserClient(); bc != nil {
		return bc
	}
	return nil
}

// directBingEnabled defaults Bing direct discovery on only for standalone
// deployments. When GO_SEARCH_URL is configured, go-search remains primary and
// Bing stays opt-in. DIRECT_BING explicitly overrides either default.
func directBingEnabled() bool {
	standalone := strings.TrimSpace(env.Str("GO_SEARCH_URL", "")) == ""
	return env.Bool("DIRECT_BING", standalone)
}

// directSearchConfig builds a search.DirectConfig from engine state.
func directSearchConfig() search.DirectConfig {
	browser := directBrowser()
	return search.DirectConfig{
		Browser:          browser,
		DDG:              browser != nil && cfg.DirectDDG,
		Startpage:        browser != nil && cfg.DirectStartpage,
		Brave:            browser != nil && cfg.DirectBrave,
		Bing:             browser != nil && directBingEnabled(),
		Reddit:           browser != nil && cfg.DirectReddit,
		Wikipedia:        browser != nil && cfg.DirectWikipedia,
		Marginalia:       browser != nil && cfg.DirectMarginalia,
		BraveLimiter:     rate.NewLimiter(1, 2),
		RedditLimiter:    rate.NewLimiter(1, 2),
		Retry:            DefaultRetryConfig,
		Metrics:          reg,
		EarlyReturnAt:    cfg.SearchEarlyReturnAt,
		PerSourceTimeout: cfg.SearchPerSourceTimeout,
	}
}

// HasDirectSearchBackend reports whether SearchDirect has at least one enabled
// web-search source and a usable browser transport. ATS discovery uses this to
// distinguish a genuine empty search from a deployment with no discovery path.
func HasDirectSearchBackend() bool {
	browser := directBrowser()
	return browser != nil && (cfg.DirectDDG || cfg.DirectStartpage || cfg.DirectBrave ||
		directBingEnabled() || cfg.DirectReddit || cfg.DirectWikipedia || cfg.DirectMarginalia)
}
