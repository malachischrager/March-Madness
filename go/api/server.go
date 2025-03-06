package api

import (
	"github.com/gin-gonic/gin"
	"net/http"
)

type Server struct {
	router *gin.Engine
}

func NewServer() *Server {
	server := &Server{
		router: gin.Default(),
	}
	server.setupRoutes()
	return server
}

func (s *Server) setupRoutes() {
	// Predictions endpoints
	s.router.GET("/api/predictions", s.getPredictions)
	s.router.GET("/api/predictions/historical", s.getHistoricalPredictions)
	
	// Team statistics endpoints
	s.router.GET("/api/teams", s.getTeams)
	s.router.GET("/api/teams/:id/stats", s.getTeamStats)
	
	// Backtesting endpoints
	s.router.POST("/api/backtest", s.runBacktest)
	s.router.GET("/api/backtest/:id", s.getBacktestResults)
}

func (s *Server) getPredictions(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"message": "Predictions endpoint",
	})
}

func (s *Server) getHistoricalPredictions(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"message": "Historical predictions endpoint",
	})
}

func (s *Server) getTeams(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"message": "Teams endpoint",
	})
}

func (s *Server) getTeamStats(c *gin.Context) {
	teamID := c.Param("id")
	c.JSON(http.StatusOK, gin.H{
		"message": "Team stats endpoint",
		"team_id": teamID,
	})
}

func (s *Server) runBacktest(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"message": "Backtest endpoint",
	})
}

func (s *Server) getBacktestResults(c *gin.Context) {
	backtestID := c.Param("id")
	c.JSON(http.StatusOK, gin.H{
		"message": "Backtest results endpoint",
		"backtest_id": backtestID,
	})
}

func (s *Server) Start(address string) error {
	return s.router.Run(address)
}
