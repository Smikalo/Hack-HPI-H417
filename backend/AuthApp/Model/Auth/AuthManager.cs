using Microsoft.AspNetCore.Authentication;
using Microsoft.AspNetCore.Authentication.Cookies;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.AspNetCore.Http.HttpResults;
using Microsoft.AspNetCore.Mvc;
using Microsoft.IdentityModel.Tokens;
using System;
using System.IdentityModel.Tokens.Jwt;
using System.Security.Claims;
using System.Text;
using System.Text.Json;

namespace SmartStartBack.Model.Auth
{
    public class AuthManager
    {
        private UserContext _userContext;
        public AuthManager(UserContext context)
        {
            _userContext = context;
        }

        public async Task<string> TryLogInAsync(User user, HttpContext _httpContext)
        {

            if (_userContext.Users.FirstOrDefault(u => u.Email == user.Email && u.Password == user.Password) == null)
            {
                var status = new StatusAnswer(false);
                // Console.WriteLine(status);
                return JsonSerializer.Serialize(status);
            }
            else
            {

                var claims = new List<Claim> { new Claim(ClaimTypes.Name, user.Email) };
                // создаем объект ClaimsIdentity
                /*
                ClaimsIdentity claimsIdentity = new ClaimsIdentity(claims, "Cookies");
                // установка аутентификационных куки
                await _httpContext.SignInAsync(CookieAuthenticationDefaults.AuthenticationScheme, new ClaimsPrincipal(claimsIdentity));
                return "Succsess";
                */
                var jwt = new JwtSecurityToken(
                    issuer: AuthOptions.ISSUER,
                    audience: AuthOptions.AUDIENCE,
                    claims: claims,
                    expires: DateTime.UtcNow.Add(TimeSpan.FromDays(365)),
                    signingCredentials: new SigningCredentials(AuthOptions.GetSymmetricSecurityKey(), SecurityAlgorithms.HmacSha256));

                //return new JwtSecurityTokenHandler().WriteToken(jwt);
                return JsonSerializer.Serialize(new TokenAnswer(new JwtSecurityTokenHandler().WriteToken(jwt), true));
            }
        }
        /*
        public async Task<string> TryLogOutAsync(HttpContext _httpContext)
        {

            await _httpContext.SignOutAsync(CookieAuthenticationDefaults.AuthenticationScheme);
            //JwtSecurityTokenHandler.
            return "Log out Succsesfully";
        }
        */
        public string TryRegister(string username, string password)
        {
            if (_userContext.Users.FirstOrDefault(u => u.Email == username) != null)
            {
                var status = new StatusAnswer(false);
                // Console.WriteLine(status);
                return JsonSerializer.Serialize(status);
            }
            else
            {
                _userContext.Add(new User(username, password));
                _userContext.SaveChanges();
                var status = new StatusAnswer(true);
                //Console.WriteLine(status);
                return JsonSerializer.Serialize(status);
            }

        }
    }
}
public class StatusAnswer
{
    public bool status { get; set; }

    public StatusAnswer(bool _status)
    {
        status = _status;
    }
    public override string ToString()
    {
        return status.ToString();
    }
}

public class TokenAnswer
{
    public bool status { get; set; }
    public string token { get; set; }
    //public string email { get; set; }

    public TokenAnswer(string token, bool _status)
    {
        this.token = token;
        status = _status;
    }
}
