using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.IdentityModel.Tokens;
using System.IdentityModel.Tokens.Jwt;
using System.Security.Claims;
using System.Text;
using System.Globalization;
using System.Text.Json;
using SmartStartBack.Model.Auth;

// For more information on enabling Web API for empty projects, visit https://go.microsoft.com/fwlink/?LinkID=397860

namespace AuthApp.Controllers
{
    [Route("auth")]
    [ApiController]
    public class AuthController : ControllerBase
    {
        private AuthManager _authManager;

        public AuthController(AuthManager authManager) {
            _authManager = authManager;
        }



        [HttpGet("{username}")]
        public User Get(string username)
        {
            var claims = new List<Claim> { new Claim(ClaimTypes.Name, username) };
            // создаем JWT-токен
            /*
            var jwt = new JwtSecurityToken(
                    issuer: AuthOptions.ISSUER,
                    audience: AuthOptions.AUDIENCE,
                    claims: claims,
                    expires: DateTime.UtcNow.Add(TimeSpan.FromMinutes(2)),
                    signingCredentials: new SigningCredentials(AuthOptions.GetSymmetricSecurityKey(), SecurityAlgorithms.HmacSha256));
            */
            //return Ok(new JwtSecurityTokenHandler().WriteToken(jwt));
            return new User(username, "pass");
        }

        [Route("register/")]
        [HttpPost]
        public string RegisterAsync([FromBody]User user)
        {
            string ans = _authManager.TryRegister(user.Email, user.Password);
            //Console.WriteLine(ans);
            return ans;
        }
        [Route("Login/")]
        [HttpPost]
        public async Task<ActionResult> LoginAsync([FromBody] User user)
        {
            return Ok(await _authManager.TryLogInAsync(user, HttpContext));
        }
        //[Authorize]
        [Route("email/")]
        [HttpGet]
        public string GetEmail()
        {
            var user = HttpContext.User;
            if(user is not null && user.Identity.IsAuthenticated)
            {
                return JsonSerializer.Serialize( new EmailAnswer(user.Identity.Name));
            }
            else
            {
                return JsonSerializer.Serialize(new StatusAnswer(false));
            }
        }
        /*
        [Route("reloadToken/")]
        [HttpPost]
        public async Task<ActionResult> ReloadToken([FromBody] string token)
        {
            return Ok(await _authManager.TryLogInAsync(user, HttpContext));
        }
        */



    }
}

public class EmailAnswer
{
    public string email { get; set; }
    public EmailAnswer(string email)
    {
        this.email = email;
    }
}
