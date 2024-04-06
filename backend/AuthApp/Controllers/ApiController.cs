using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using System.Text.Json;
using SmartStartBack.Model.Path;
using SmartStartBack.Model;

// For more information on enabling Web API for empty projects, visit https://go.microsoft.com/fwlink/?LinkID=397860

namespace AuthApp.Controllers
{
    [Route("api")]
    //[Authorize]
    [ApiController]
    public class ApiController : ControllerBase
    {
        private IPathFinder _pathFinder;
        public ApiController(IPathFinder pf) 
        {
            _pathFinder = pf;
        }
        // GET: api/<ValuesController>
      //  [Route("test/")]
      //  [HttpGet]
       // public async Task<string> Get()
      //  {
            //var path =await  _pathFinder.GetShortestPath(new MapPoint[] { new MapPoint(39.96335687073061, -86.13645241042872), new MapPoint(39.97403352477058, -86.10513864849375) });
            //return JsonSerializer.Serialize(new PathRequest(path, true));
      //  }
        [Route ("path")]
        [HttpPost]
        public async Task<string> GetPath([FromBody] PathRequest pathReq) {
            //if (pathReq.isSmooth)
            //{
                //return JsonSerializer.Serialize(_pathFinder.GetSmoothestPath(pathReq.startEnd));
            //}
            //else
            //{
                return JsonSerializer.Serialize(await _pathFinder.GetShortestPath(pathReq.startEnd));
            //}
        }

    }
}
